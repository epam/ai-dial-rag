import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from aidial_rag.app import create_app
from aidial_rag.app_config import AppConfig
from tests.utils.e2e_decorator import e2e_test

MIDDLEWARE_HOST = "http://localhost:8081"


def _make_test_request(question, attachments) -> str:
    app = create_app(app_config=AppConfig(dial_url=MIDDLEWARE_HOST))
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
    return json_response["choices"][0]["message"]["content"]


@pytest.mark.asyncio
@e2e_test(filenames=["test_file.csv"])
async def test_document_error(attachments, caplog):
    message = _make_test_request("What is this csv about?", attachments)
    assert (
        "I'm sorry, but I can't process the documents because of the following errors:\n\n"
        "|Document|Error|\n"
        "|---|---|\n"
        "|test_file.csv|Unable to load document content. Try another document format.|\n\n"
        "Please try again with different documents." in message
    )

    assert "test_file.csv" not in caplog.text
    assert "Parser:  Parsing document started" in caplog.text

    assert (
        "aidial_rag.errors.DocumentProcessingError: Error on processing document: "
        "unhandled errors in a TaskGroup (1 sub-exception)" in caplog.text
    )
    assert (
        "aidial_rag.errors.InvalidDocumentError: Unable to load document content. "
        "Try another document format." in caplog.text
    )


@pytest.mark.asyncio
@e2e_test(filenames=["test_file.csv"])
async def test_document_error_with_error_log_enabled(attachments, caplog):
    with patch.dict(
        "os.environ",
        {
            "DIAL_RAG__REQUEST__LOG_DOCUMENT_LINKS": "true",
        },
    ):
        message = _make_test_request("What is this csv about?", attachments)

    assert (
        "I'm sorry, but I can't process the documents because of the following errors:\n\n"
        "|Document|Error|\n"
        "|---|---|\n"
        "|test_file.csv|Unable to load document content. Try another document format.|\n\n"
        "Please try again with different documents." in message
    )

    assert "test_file.csv" in caplog.text
    assert (
        "<files/6iTkeGUs2CvUehhYLmMYXB/test_file.csv>:  Parser:  Parsing document started"
        in caplog.text
    )

    assert (
        "aidial_rag.errors.DocumentProcessingError: Error on processing document "
        "files/6iTkeGUs2CvUehhYLmMYXB/test_file.csv: "
        "unhandled errors in a TaskGroup (1 sub-exception)" in caplog.text
    )
    assert (
        "aidial_rag.errors.InvalidDocumentError: Unable to load document content. "
        "Try another document format." in caplog.text
    )


@pytest.mark.asyncio
@e2e_test(filenames=["test_file.csv"])
async def test_wrong_filename(attachments, caplog):
    attachments[0]["url"] = attachments[0]["url"].replace(".csv", ".xls")
    message = _make_test_request("What is this csv about?", attachments)

    assert (
        "I'm sorry, but I can't process the documents because of the following errors:\n\n"
        "|Document|Error|\n"
        "|---|---|\n"
        "|test_file.xls|404 Not Found|\n\n"
        "Please try again with different documents." in message
    )

    assert "test_file.xls" not in caplog.text
    assert "Parser:  Parsing document started" not in caplog.text

    assert (
        "aidial_rag.errors.DocumentProcessingError: Error on processing document: 404 Not Found"
        in caplog.text
    )
    assert (
        "aidial_rag.errors.InvalidDocumentError: 404 Not Found" in caplog.text
    )


@pytest.mark.asyncio
@e2e_test(filenames=["test_file.csv"])
async def test_wrong_filename_with_error_log_enabled(attachments, caplog):
    with patch.dict(
        "os.environ",
        {
            "DIAL_RAG__REQUEST__LOG_DOCUMENT_LINKS": "true",
        },
    ):
        attachments[0]["url"] = attachments[0]["url"].replace(".csv", ".xls")
        message = _make_test_request("What is this csv about?", attachments)

    assert (
        "I'm sorry, but I can't process the documents because of the following errors:\n\n"
        "|Document|Error|\n"
        "|---|---|\n"
        "|test_file.xls|404 Not Found|\n\n"
        "Please try again with different documents." in message
    )

    assert "test_file.xls" in caplog.text
    assert "Parser:  Parsing document started" not in caplog.text

    assert (
        "aidial_rag.errors.DocumentProcessingError: Error on processing document "
        "files/6iTkeGUs2CvUehhYLmMYXB/test_file.xls: 404 Not Found"
        in caplog.text
    )
    assert (
        "aidial_rag.errors.InvalidDocumentError: 404 Not Found" in caplog.text
    )
