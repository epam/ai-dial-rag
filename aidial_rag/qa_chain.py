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
    AIMessage,
    BaseMessage,
    HumanMessage,
    merge_content,
)
from langchain_core.runnables import chain
from pydantic import Field

from aidial_rag.aidial_to_langchain import to_langchain_messages
from aidial_rag.base_config import BaseConfig
from aidial_rag.document_record import DocumentRecord
from aidial_rag.image_processor.base64 import pil_image_as_base64
from aidial_rag.image_processor.extract_pages import (
    are_image_pages_supported,
    extract_pages_gen,
)
from aidial_rag.index_record import ChunkMetadata, RetrievalType
from aidial_rag.llm import LlmConfig, create_llm
from aidial_rag.query_chain import QueryChainConfig, create_get_query_chain
from aidial_rag.request_context import RequestContext

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

REF_HISTORY_PATTERN = re.compile(r"\[(\d+)\]")

INCLUDED_ATTRIBUTES = ["source", "page_number", "title"]


class ChatChainConfig(BaseConfig):
    llm: LlmConfig = Field(
        default=LlmConfig(),
        description=(
            "Configuration for the LLM used in the chat chain. "
            "The model should support vision if `num_page_images_to_use` is greater than 0."
        ),
    )
    system_prompt_template_override: str | None = Field(
        default=None,
        description="Allow to override the system prompt template.",
    )
    use_history: bool = Field(
        default=True,
        description=(
            "Used to set whether to use the history for the answer generation. "
            "If true, the previous messages from the chat history would be passes to the model. "
            "If false, only the query (last user message or standalone question, depending on the "
            "`query_chain` settings) will be passed to the model for the answer generation."
        ),
    )
    num_page_images_to_use: int = Field(
        default=4,
        description=(
            "Sets number of page images to pass to the model for the answer generation. "
            "If is greater that 0, the model in `llm.deployment_name` should accept images "
            "in the user messages. Could be set to 0 (together with USE_MULTIMODAL_INDEX=False "
            "and USE_DESCRIPTION_INDEX=False) for text-only RAG."
        ),
    )
    page_image_size: int = Field(
        default=1536,
        description="Sets the size of the page images to pass to the model for the answer generation.",
    )


class QAChainConfig(BaseConfig):
    chat_chain: ChatChainConfig = Field(
        default=ChatChainConfig(),
        description=(
            "Configuration for the chat chain which generates the answer for the user question "
            "based on the retrieved context."
        ),
    )
    query_chain: QueryChainConfig = Field(
        default=QueryChainConfig(),
        description=(
            "Configuration for the query chain which reformulates the user question to standalone "
            "question for the retrieval based on the chat history."
        ),
    )


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


def collect_pages_with_images(
    doc_records: List[DocumentRecord], chunks_metadatas
):
    # RetrievalType.IMAGE has higher priority
    for chunk_metadata in chunks_metadatas:
        doc_record = doc_records[chunk_metadata["doc_id"]]
        if not are_image_pages_supported(doc_record.mime_type):
            continue
        chunk = doc_record.chunks[chunk_metadata["chunk_id"]]
        if (
            chunk_metadata["retrieval_type"] == RetrievalType.IMAGE
            and "page_number" in chunk.metadata
        ):
            yield (chunk_metadata["doc_id"], chunk.metadata["page_number"])

    for chunk_metadata in chunks_metadatas:
        doc_record = doc_records[chunk_metadata["doc_id"]]
        if not are_image_pages_supported(doc_record.mime_type):
            continue
        chunk = doc_record.chunks[chunk_metadata["chunk_id"]]
        if (
            chunk_metadata["retrieval_type"] != RetrievalType.IMAGE
            and "page_number" in chunk.metadata
        ):
            yield (chunk_metadata["doc_id"], chunk.metadata["page_number"])


async def make_image_by_page(
    doc_records: List[DocumentRecord],
    chunks_metadatas,
    num_pages_to_use: int,
    page_image_size: int,
) -> dict:
    required_pages = set()
    for doc_id, page_number in collect_pages_with_images(
        doc_records, chunks_metadatas
    ):
        if len(required_pages) >= num_pages_to_use:
            break
        required_pages.add((doc_id, page_number))

    image_by_page = {}
    for doc_id, pages_iter in groupby(sorted(required_pages), itemgetter(0)):
        page_numbers = [page_number for _, page_number in pages_iter]
        doc_record = doc_records[doc_id]
        page_images_gen = extract_pages_gen(
            doc_record.mime_type,
            doc_record.document_bytes,
            page_numbers,
            scaled_size=page_image_size,
        )
        page_numbers_it = iter(page_numbers)
        async for page_image in page_images_gen:
            image_by_page[doc_id, next(page_numbers_it)] = pil_image_as_base64(
                page_image, format="PNG"
            )

    return image_by_page


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
    chunks_metadatas = [
        ChunkMetadata(**index_item.metadata) for index_item in index_items
    ]

    image_by_page = await make_image_by_page(
        doc_records,
        chunks_metadatas,
        config.num_page_images_to_use,
        config.page_image_size,
    )

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


def transform_history_message(message: BaseMessage) -> BaseMessage:
    if isinstance(message, AIMessage) and message.content:
        # Restore the references to <[num]> in the assistant messages, because
        # the model may be confused if the format is different from the prompt
        return AIMessage(
            content=REF_HISTORY_PATTERN.sub(r"<[\1]>", str(message.content))
        )

    return message


def transform_history(messages: List[Message]) -> List[BaseMessage]:
    return [
        transform_history_message(message)
        for message in to_langchain_messages(messages)
        if message.content
    ]


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
