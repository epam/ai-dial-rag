import logging
import re
from typing import Any, AsyncIterator, Callable, Dict, List, cast

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    merge_content,
)
from langchain_core.runnables import Runnable, chain

from aidial_rag.llm import create_llm
from aidial_rag.qa_chain_config import ChatChainConfig
from aidial_rag.request_context import RequestContext
from aidial_rag.retrieval_api import RetrievalResults

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

INCLUDED_ATTRIBUTES = ["page_number", "source", "title"]


def format_attributes(i, metadata: Dict[str, int | str]) -> str:
    attributes = [("id", i)] + [
        (k, metadata[k]) for k in INCLUDED_ATTRIBUTES if k in metadata
    ]
    return " ".join(f"{k}='{v}'" for k, v in attributes)


MessageElement = Dict[str, str | Dict[str, str]]


def text_element(text: str) -> MessageElement:
    return {"type": "text", "text": text}


def image_element(image: str) -> MessageElement:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image}"},
    }


def create_docs_message(
    retrieval_results: RetrievalResults,
) -> List[MessageElement]:
    docs_message: List[MessageElement] = []
    docs_message.append(text_element("<context>"))
    for i, chunk in enumerate(retrieval_results.chunks, start=1):
        attributes = format_attributes(i, chunk.model_dump(exclude_none=True))
        docs_message.append(text_element(f"<doc {attributes}>\n{chunk.text}\n"))

        if chunk.page_image_index is not None:
            image = retrieval_results.images[chunk.page_image_index]
            docs_message.append(image_element(image.data))

        docs_message.append(text_element("</doc>\n"))
    docs_message.append(text_element("</context>"))

    return docs_message


# The chain should be async, because otherwise langchain would run it in a threadpool
# and we may get several chains running in parallel.
# Some functions used in chain may be not thread-safe, like extract_pages_gen
@chain
async def create_chat_prompt(input: Dict[str, Any]) -> List[BaseMessage]:
    config: ChatChainConfig = input["chat_chain_config"]

    system_prompt_template = (
        config.system_prompt_template_override or DEFAULT_SYSTEM_TEMPLATE
    )

    retrieval_results = input["retrieval_results"]
    docs_message = create_docs_message(retrieval_results)

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
    retrieval_chain: Runnable[Dict[str, Any], Dict[str, Any]],
    chain_input: Dict[str, Any],
    content_callback: Callable[[str], None],
) -> List[Document]:
    chat_chain_config = chain_input["chat_chain_config"]

    llm = create_llm(request_context.dial_config, chat_chain_config.llm)
    qa_chain = retrieval_chain.assign(
        answer=create_chat_prompt | llm | StrOutputParser()
    ).pick(["found_items", "answer"])

    reference_items = []
    async for r in get_reference_documents(chain_input, qa_chain):
        if "answer" in r:
            content_callback(r["answer"])
        if "reference_items" in r:
            reference_items = r["reference_items"]

    return reference_items
