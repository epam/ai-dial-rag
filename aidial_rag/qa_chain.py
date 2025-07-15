import logging
import re
from itertools import groupby
from operator import itemgetter
from typing import AsyncIterator, Callable, Dict, List, cast

from aidial_sdk.chat_completion import Message
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import BaseRetriever, Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.messages import (
    HumanMessage,
    merge_content,
)
from langchain_core.runnables import chain

from aidial_rag.document_record import DocumentRecord
from aidial_rag.index_record import ChunkMetadata
from aidial_rag.llm import create_llm
from aidial_rag.qa_chain_config import ChatChainConfig, QAChainConfig
from aidial_rag.query_chain import create_get_query_chain
from aidial_rag.request_context import RequestContext
from aidial_rag.retrieval_chain import create_image_by_page
from aidial_rag.transform_history import transform_history

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_TEMPLATE = """You are helpful assistant. You are to answer the user questions based on user provided documents.
User can attach the documents to the conversation by using the paperclip button.
The attachments are already processed by the system and the relevant pieces of the documents are available in the context.
The pdf, doc, ppt and text files are supported for the attachments.
Use the following pieces of context from user documents and the images of the pages from user documents to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Current date is _date_.

Anything between the 'context' xml blocks is retrieved from a knowledge bank, not part of the conversation with the user.

Cite pieces of context using <[number]> notation (like <[2]>). Only cite the most relevant pieces of context that answer the question accurately.
Place these citations at the end of the sentence or paragraph that reference them - do not put them all at the end.
If different citations refer to different entities within the same name, write separate answers for each entity.
If you want to cite multiple pieces of context for the same sentence, format it as `<[number1]> <[number2]>`.
However, you should NEVER do this with the same number - if you want to cite `number1` multiple times for a sentence, only do `<[number1]>` not `<[number1]> <[number1]>`.
"""


SINGLE_QUERY_TEMPLATE = HumanMessagePromptTemplate.from_template("{query}")

REF_PATTERN = re.compile(r"<\[(\d+)\]>")

INCLUDED_ATTRIBUTES = ["source", "page_number", "title"]


def format_attributes(i, metadata: dict) -> str:
    attributes = [("id", i)] + [
        (k, v) for k, v in metadata.items() if k in INCLUDED_ATTRIBUTES
    ]
    return " ".join(f"{k}='{v}'" for k, v in attributes)


def text_element(text: str) -> dict:
    return {"type": "text", "text": text}


def image_element(image: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image}"},
    }


def create_docs_message(
    doc_records, chunks_metadatas, image_by_page
) -> List[Dict[str, dict]]:
    attached_images = set()
    docs_message = []
    docs_message.append(text_element("<context>"))
    for i, chunk_metadata in enumerate(chunks_metadatas, start=1):
        doc_record = doc_records[chunk_metadata["doc_id"]]
        chunk = doc_record.chunks[chunk_metadata["chunk_id"]]

        attributes = format_attributes(i, chunk.metadata)
        docs_message.append(text_element(f"<doc {attributes}>\n{chunk.text}\n"))

        image_key = (
            chunk_metadata["doc_id"],
            chunk.metadata.get("page_number"),
        )
        if image_key not in attached_images and (
            image := image_by_page.get(image_key)
        ):
            docs_message.append(image_element(image))
            attached_images.add(image_key)

        docs_message.append(text_element("</doc>\n"))
    docs_message.append(text_element("</context>"))

    return docs_message


# The chain should be async, because otherwise langchain would run it in a threadpool
# and we may get several chains running in parallel.
# Some functions used in chain may be not thread-safe, like extract_pages_gen
@chain
async def create_chat_prompt(input: dict):
    config: ChatChainConfig = input["chat_chain_config"]

    system_prompt_template = (
        config.system_prompt_template_override or DEFAULT_SYSTEM_TEMPLATE
    )

    doc_records: List[DocumentRecord] = input.get("doc_records", [])
    index_items: List[Document] = input.get("found_items", [])
    image_by_page: Dict[tuple, str] = input.get("image_by_page", {})

    chunks_metadatas = [
        ChunkMetadata(**index_item.metadata) for index_item in index_items
    ]

    docs_message = create_docs_message(
        doc_records, chunks_metadatas, image_by_page
    )

    template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt_template),
            MessagesPlaceholder("chat_history")
            if config.use_history
            else SINGLE_QUERY_TEMPLATE,
        ]
    )

    prompt_messages = template.invoke(input).to_messages()
    assert len(prompt_messages) > 1
    last_message = prompt_messages[-1]
    assert isinstance(last_message, HumanMessage)
    assert isinstance(last_message.content, str)

    # Need cast here, because list is mutable container and List[Dict] is not accepted as List[str | Dict]
    # https://github.com/microsoft/pyright/blob/main/docs/type-concepts.md#generic-types
    merged_content = merge_content(
        cast(List[str | Dict], [text_element(last_message.content)]),
        cast(List[str | Dict], docs_message),
    )

    prompt_messages[-1] = HumanMessage(content=merged_content)
    return prompt_messages


# TODO: Rewrite this function to be a chain and be able to work with pipe operator
async def get_reference_documents(chain_input, chain) -> AsyncIterator:  # noqa: C901
    used_doc_ids = []
    # Variable to catch pieces of document links in different chunks, like this
    # "first chunk <["; "1]> second chunk"
    prev_piece = ""
    found_items = None
    async for r in chain.astream(chain_input):
        if "found_items" in r:
            found_items = r["found_items"]

        if "answer" in r:
            answer_piece = prev_piece + r["answer"]
            last_pos = 0
            for m in REF_PATTERN.finditer(answer_piece):
                chunk_id = int(m.group(1))
                # FIXME: hotfix for cases, when there is link
                # inside of document content, like [23]
                assert found_items is not None
                if not (1 <= chunk_id <= len(found_items)):
                    logger.warning(
                        "Chunk ID in model response is out of bounds:"
                        f"{chunk_id} / {len(found_items)}"
                    )
                    yield {"answer": answer_piece[last_pos : m.end()]}
                    last_pos = m.end()
                    continue

                # id in model response is starting from 1
                chunk_index = chunk_id - 1
                if chunk_index not in used_doc_ids:
                    used_doc_ids.append(chunk_index)
                reference_index = used_doc_ids.index(chunk_index)
                yield {
                    "answer": answer_piece[last_pos : m.start()]
                    + f"[{reference_index + 1}]"
                }
                last_pos = m.end()

            pos = answer_piece.find("<[", last_pos)
            if pos == -1:
                if answer_piece and answer_piece[-1] == "<":
                    pos = len(answer_piece) - 1
                else:
                    pos = len(answer_piece)
            yield {"answer": answer_piece[last_pos:pos]}
            prev_piece = answer_piece[pos:]
    if prev_piece:
        yield {"answer": prev_piece}

    if found_items is not None:
        reference_items = [found_items[i] for i in used_doc_ids]
        yield {"reference_items": reference_items}


async def generate_answer(
    request_context: RequestContext,
    qa_chain_config: QAChainConfig,
    retriever: BaseRetriever,
    messages: List[Message],
    content_callback: Callable[[str], None],
    document_records: List[DocumentRecord],
) -> List[Document]:
    llm = create_llm(
        request_context.dial_config, qa_chain_config.chat_chain.llm
    )

    get_query_chain = create_get_query_chain(
        request_context, qa_chain_config.query_chain
    )

    qa_chain = (
        RunnablePassthrough()
        .assign(query=get_query_chain)
        .assign(found_items=(itemgetter("query") | retriever))
        .assign(image_by_page=create_image_by_page)
        .assign(answer=(create_chat_prompt | llm | StrOutputParser()))
    ).pick(["found_items", "answer"])

    chain_input = {
        # We may have empty messages after command processing
        # Some models (like claude) do not support empty messages
        "chat_history": transform_history(messages),
        "chat_chain_config": qa_chain_config.chat_chain,
        "doc_records": document_records,
    }

    reference_items = []
    async for r in get_reference_documents(chain_input, qa_chain):
        if "answer" in r:
            content_callback(r["answer"])
        if "reference_items" in r:
            reference_items = r["reference_items"]

    return reference_items
