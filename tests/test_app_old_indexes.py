import pytest
from fastapi.testclient import TestClient

from aidial_rag.app import create_app
from aidial_rag.app_config import AppConfig
from aidial_rag.index_mime_type import INDEX_MIME_TYPE
from aidial_rag.index_storage import IndexStorageConfig
from aidial_rag.retrieval_api import Page, RetrievalResponse
from tests.utils.config_override import (
    description_index_retries_override,  # noqa: F401
)
from tests.utils.e2e_decorator import e2e_test
from tests.utils.response_helpers import (
    get_attachments,
    get_retrieval_response_json,
)

middleware_host = "http://localhost:8081"

pytestmark = pytest.mark.usefixtures("description_index_retries_override")


# e2e_test is not compatible with pytest.mark.parametrize,
# so we create separate test functions for each index file.
async def do_request_with_old_index(attachments, index_file):
    app = create_app(
        app_config=AppConfig(
            dial_url=middleware_host,
            index_storage=IndexStorageConfig(use_dial_file_storage=True),
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
                    "content": "What is the highest peak in the Alps?",
                    "custom_content": {
                        "attachments": attachments
                        + [
                            {
                                "type": INDEX_MIME_TYPE,
                                "url": f"files/6iTkeGUs2CvUehhYLmMYXB/index/{index_file}",
                                "reference_url": attachments[0]["url"],
                            }
                        ]
                    },
                }
            ],
            "custom_fields": {
                "configuration": {
                    "request": {
                        "type": "retrieval",
                        "allow_indexing": False,
                        "save_index_on_migration": False,
                    }
                }
            },
        },
        timeout=60.0,
    )

    assert response.status_code == 200
    retrieval_response_attachments = get_attachments(response.json())
    retrieval_response = RetrievalResponse.model_validate(
        get_retrieval_response_json(retrieval_response_attachments)
    )
    assert retrieval_response.chunks[0].text is not None
    assert "Mont Blanc" in retrieval_response.chunks[0].text

    assert retrieval_response.chunks[0].page == Page(
        number=1,
        image_index=0,
    )


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.pdf"])
async def test_old_index_format_11(attachments):
    await do_request_with_old_index(attachments, "doc_record_0.22.0.bin")


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.pdf"])
async def test_old_index_format_12(attachments):
    await do_request_with_old_index(attachments, "doc_record_0.33.0.bin")


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.pdf"])
async def test_old_index_format_13_before_modification_metadata(attachments):
    await do_request_with_old_index(attachments, "doc_record_0.34.0rc0.bin")
