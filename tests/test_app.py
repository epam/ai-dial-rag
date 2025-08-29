import json

import pytest
from fastapi.testclient import TestClient

from aidial_rag.app import create_app
from aidial_rag.app_config import AppConfig
from tests.utils.config_override import (
    description_index_retries_override,  # noqa: F401
)
from tests.utils.e2e_decorator import e2e_test

middleware_host = "http://localhost:8081"

pytestmark = pytest.mark.usefixtures("description_index_retries_override")


def check_expected_text(expected_text, actual_text):
    """
    Due to LLM response entropy, the response may contain different synonyms of same term
    The function able to check if a string contains any of synonyms from expected_text in actual text, divided by "|"
    character. For example, for the question "What's this document about?", LLM might give different right answers:
    - "Overall, ... "
    - "... it's a compilation of ...
    - "Document is an overview of ..."

    For such case, it's useful to pass a string like "overall|compilation|overview", that would ensure that answer
    contains at least one of these words, to mark check as passed.
    """
    actual_text_lower = actual_text.lower()
    if "|" in expected_text:
        string_list = expected_text.split("|")
        assert any(
            string.lower() in actual_text_lower for string in string_list
        )
    else:
        assert expected_text.lower() in actual_text_lower


def run_simple_test(attachments, question, expected_text_list):
    app = create_app(
        app_config=AppConfig(
            dial_url=middleware_host,
            enable_debug_commands=True,
        )
    )
    client = TestClient(app)
    response = client.post(
        "/openai/deployments/dial-rag/chat/completions",
        headers={"Api-Key": "api-key"},
        json={
            "model": "dial-rag",
            "messages": [
                {
                    "role": "user",
                    "content": question,
                    "custom_content": {"attachments": attachments},
                }
            ],
        },
        timeout=100.0,
    )

    assert response.status_code == 200
    json_response = json.loads(response.text)

    for expected_text in expected_text_list:
        check_expected_text(
            expected_text, json_response["choices"][0]["message"]["content"]
        )

    return json_response


@pytest.mark.skip
@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.pdf"])
async def test_chat_completion(attachments):
    run_simple_test(
        attachments,
        "What length of the Alps?",
        ["1200 kilometers|1,200 kilometers|1,200 km|1200 km|750 mi|750 miles"],
    )


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.html"])
async def test_chat_completion_gemini(attachments):
    run_simple_test(
        attachments,
        "/model gemini-1.5-pro-002\nWhat is the highest peak in the Alps?",
        ["Mont Blanc"],
    )


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.html"])
async def test_chat_completion_history(attachments):
    app = create_app(
        app_config=AppConfig(
            dial_url=middleware_host,
        )
    )
    client = TestClient(app)
    response = client.post(
        "/openai/deployments/dial-rag/chat/completions",
        headers={"Api-Key": "api-key"},
        json={
            "model": "dial-rag",
            "messages": [
                {
                    "role": "user",
                    "content": "What is Alps?",
                    "custom_content": {"attachments": attachments},
                },
                {
                    "role": "assistant",
                    "content": "The Alps are a major mountain range system located in Europe, "
                    "stretching across eight countries: Austria, France, Germany, "
                    "Italy, Liechtenstein, Monaco, Slovenia, and Switzerland.",
                },
                {
                    "role": "user",
                    "content": "What is the highest peak in the Alps?",
                },
            ],
        },
        timeout=100.0,
    )

    assert response.status_code == 200
    json_response = json.loads(response.text)

    check_expected_text(
        "Mont Blanc", json_response["choices"][0]["message"]["content"]
    )


@pytest.mark.asyncio
@e2e_test()
async def test_chat_completion_simple():
    app = create_app(
        app_config=AppConfig(
            dial_url=middleware_host,
        )
    )
    client = TestClient(app)
    response = client.post(
        "/openai/deployments/dial-rag/chat/completions",
        headers={"Api-Key": "api-key"},
        json={
            "model": "dial-rag",
            "messages": [{"role": "user", "content": "Hi!"}],
        },
        timeout=60.0,
    )

    assert response.status_code == 200
    json_response = json.loads(response.text)
    expected_text = "Hello! How can I assist you today?"
    assert expected_text in json_response["choices"][0]["message"]["content"]


@pytest.mark.asyncio
@e2e_test(filenames=["test_pdf_with_image_and_text.pdf"])
async def test_chat_completion_image_and_text(attachments):
    run_simple_test(attachments, "What is on the image?", ["pencil"])


@pytest.mark.asyncio
@e2e_test(filenames=["test_image.png"])
async def test_chat_completion_image_png(attachments):
    run_simple_test(
        attachments, "What is the shape of the infographic?", ["pencil"]
    )


@pytest.mark.asyncio
@e2e_test(filenames=["test_image.tiff"])
async def test_chat_completion_image_tiff(attachments):
    run_simple_test(
        attachments, "What is the shape of the infographic?", ["pencil"]
    )


@pytest.mark.asyncio
@e2e_test(filenames=["test_image.jpg"])
async def test_chat_completion_image_jpg(attachments):
    run_simple_test(
        attachments, "What is the shape of the infographic?", ["pencil"]
    )


@pytest.mark.asyncio
@e2e_test(filenames=["test_image.bmp"])
async def test_chat_completion_image_bmp(attachments):
    run_simple_test(
        attachments, "What is the shape of the infographic?", ["pencil"]
    )


@pytest.mark.asyncio
@e2e_test()
async def test_model_command():
    app = create_app(
        app_config=AppConfig(
            dial_url=middleware_host,
            enable_debug_commands=True,
        )
    )
    client = TestClient(app)
    response = client.post(
        "/openai/deployments/dial-rag/chat/completions",
        headers={"Api-Key": "api-key"},
        json={
            "model": "dial-rag",
            "messages": [
                {"role": "user", "content": "/model gemini-1.5-flash-001\nHi!"}
            ],
        },
        timeout=60.0,
    )

    assert response.status_code == 200
    json_response = json.loads(response.text)
    expected_text = "How can I help you today?"
    assert expected_text in json_response["choices"][0]["message"]["content"]


@pytest.mark.asyncio
@e2e_test(filenames=["test_file.csv"])
async def test_csv_with_different_number_of_columns(attachments):
    run_simple_test(
        attachments,
        "What is this csv about?",
        ["Unable to load document content. Try another document format."],
    )


@pytest.mark.asyncio
@e2e_test(filenames=["test_presentation.pptx"])
async def test_presentation_pptx(attachments):
    run_simple_test(
        attachments,
        "What are the nodes connected to the System on the chart drawn by hand?",
        ["plugins", "addons"],
    )
    run_simple_test(
        attachments,
        "What is the number of sales for the quarter with the highest sales?",
        ["8.2"],
    )


@pytest.mark.asyncio
@e2e_test(filenames=["test_presentation.ppt"])
async def test_presentation_ppt(attachments):
    run_simple_test(
        attachments,
        "What are the nodes connected to the System on the chart drawn by hand?",
        ["plugins", "addons"],
    )
    run_simple_test(
        attachments,
        "What is the number of sales for the quarter with the highest sales?",
        ["8.2"],
    )


@pytest.mark.asyncio
@e2e_test(filenames=["test_presentation.pptm"])
async def test_presentation_pptm(attachments):
    run_simple_test(
        attachments,
        "What are the nodes connected to the System on the chart drawn by hand?",
        ["plugins", "addons"],
    )
    run_simple_test(
        attachments,
        "What is the number of sales for the quarter with the highest sales?",
        ["8.2"],
    )


@pytest.mark.asyncio
@e2e_test(filenames=["test_presentation.odp"])
async def test_presentation_odp(attachments):
    run_simple_test(
        attachments,
        "What are the nodes connected to the System on the chart drawn by hand?",
        ["plugins", "addons"],
    )
    run_simple_test(
        attachments,
        "What is the number of sales for the quarter with the highest sales?",
        ["8.2"],
    )


@pytest.mark.asyncio
@e2e_test(filenames=["test_presentation.docx"])
async def test_presentation_docx(attachments):
    run_simple_test(
        attachments,
        "What are the nodes connected to the System on the chart drawn by hand?",
        ["plugins", "addons"],
    )
    run_simple_test(
        attachments,
        "What is the number of sales for the quarter with the highest sales?",
        ["8.2"],
    )


@pytest.mark.asyncio
@e2e_test(filenames=["test_presentation.doc"])
async def test_presentation_doc(attachments):
    run_simple_test(
        attachments,
        "What are the nodes connected to the System on the chart drawn by hand?",
        ["plugins", "addons"],
    )
    run_simple_test(
        attachments,
        "What is the number of sales for the quarter with the highest sales?",
        ["8.2"],
    )


@pytest.mark.asyncio
@e2e_test(filenames=["test_presentation.odt"])
async def test_presentation_odt(attachments):
    run_simple_test(
        attachments,
        "What are the nodes connected to the System on the chart drawn by hand?",
        ["plugins", "addons"],
    )
    run_simple_test(
        attachments,
        "What is the number of sales for the quarter with the highest sales?",
        ["8.2"],
    )


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.html", "test_image.png"])
async def test_mix_of_image_and_non_image_formats(attachments):
    json_response = run_simple_test(
        attachments, "Where is an infographic with a pencil?", ["pencil"]
    )

    result_attachments = json_response["choices"][0]["message"][
        "custom_content"
    ]["attachments"]
    assert len(result_attachments) == 1
    assert result_attachments[0]["title"] == "[1] test_image.png"
