from operator import itemgetter

import pytest
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.chat_models import FakeListChatModel

from aidial_rag.qa_chain import get_reference_documents


async def run_qa_chain(documents, response, query="query for fake model"):
    qa_chain_mock = RunnablePassthrough().assign(
        answer=(
            itemgetter("query")
            | FakeListChatModel(responses=[response])
            | StrOutputParser()
        )
    )

    chain_input = {
        "found_items": documents,
        "query": query,
    }

    result_chunks = [
        result_chunk
        async for result_chunk in get_reference_documents(
            chain_input, qa_chain_mock
        )
    ]

    result_documents = next(
        chunk["reference_items"]
        for chunk in result_chunks
        if "reference_items" in chunk
    )

    answer_chunks = [chunk for chunk in result_chunks if "answer" in chunk]
    # Check that streaming works correctly
    assert len(answer_chunks) > 1

    total_content = "".join(chunk["answer"] for chunk in answer_chunks)
    return total_content, result_documents


@pytest.mark.asyncio
async def test_answer_with_links_in_text():
    """
    [6] and [5] are links from the model to the documents
    [12] is link from document text, that we should ignore
    """

    documents = [
        Document(
            page_content="Useless content 1",
        ),
        Document(
            page_content="Useless content 2",
        ),
        Document(
            page_content="Useless content 3",
        ),
        Document(
            page_content="Useless content 4",
        ),
        Document(
            page_content="<[12]> Interesting article",
        ),
        Document(page_content="Other article without link"),
    ]

    model_response = (
        "The document is about Other article without link <[6]> "
        "and <[12]> Interesting article <[5]>"
    )

    total_content, result_documents = await run_qa_chain(
        documents=documents, response=model_response
    )

    # Document ids were converted to indexes
    assert total_content == (
        "The document is about Other article without link [1] "
        "and <[12]> Interesting article [2]"
    )

    # Correct documents were returned
    assert len(result_documents) == 2
    assert result_documents[0] == documents[5]
    assert result_documents[1] == documents[4]


@pytest.mark.asyncio
async def test_answer_with_single_attachment():
    """
    Test with a single document attached, to check the border case
    """

    documents = (
        [
            Document(page_content="Single document"),
        ],
    )

    total_content, result_documents = await run_qa_chain(
        documents=documents,
        response="The answer is in a single document <[1]>",
    )

    assert total_content == "The answer is in a single document [1]"
    assert len(result_documents) == 1
    assert result_documents[0] == documents[0]


@pytest.mark.asyncio
async def test_answer_chunks():
    """
    Test with a single document attached, to check the chunks boundaries
    """

    documents = (
        [
            Document(page_content="Single document"),
        ],
    )

    response = "The answer is in a single document <[1]>"

    for i in range(1, len(response)):
        response_chunks = [response[:i], response[i:]]

        total_content, result_documents = await run_qa_chain(
            documents=documents,
            response=response_chunks,
        )

        assert total_content == "The answer is in a single document [1]"
        assert len(result_documents) == 1
        assert result_documents[0] == documents[0]
