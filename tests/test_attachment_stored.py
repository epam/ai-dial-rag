from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aidial_sdk.chat_completion import Choice, Stage
from pydantic import SecretStr

from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.configuration_endpoint import RequestConfig
from aidial_rag.dial_api_client import DialApiClient
from aidial_rag.document_loaders import load_attachment
from aidial_rag.document_record import DocumentRecord
from aidial_rag.documents import load_document
from aidial_rag.errors import DocumentProcessingError, InvalidDocumentError
from aidial_rag.index_storage import IndexStorageHolder, link_to_index_url
from aidial_rag.indexing_config import IndexingConfig
from aidial_rag.indexing_task import IndexingTask
from aidial_rag.request_context import RequestContext
from aidial_rag.resources.dial_limited_resources import DialLimitedResources
from aidial_rag.retrievers.colpali_retriever.colpali_model_resource import (
    ColpaliModelResource,
)
from tests.utils.user_limits_mock import user_limits_mock

request_config = RequestConfig(
    indexing=IndexingConfig(multimodal_index=None, description_index=None),
)


@pytest.fixture
def request_context():
    return RequestContext(
        dial_url="http://localhost:8080",
        api_key=SecretStr("ABRAKADABRA"),
        choice=Choice(queue=MagicMock(), choice_index=0),
        dial_limited_resources=DialLimitedResources(user_limits_mock()),
    )


class MockDialApiClient(DialApiClient):
    def __init__(self):
        self.bucket_id = "test_bucket"
        self.storage = {}

    async def get_file(self, relative_url):
        if relative_url in self.storage:
            return self.storage[relative_url]
        return None

    async def put_file(self, relative_url, data, content_type):
        self.storage[relative_url] = data
        return {}


@pytest.fixture
def dial_api_client():
    return MockDialApiClient()


@pytest.fixture
def index_storage(dial_api_client):
    return IndexStorageHolder().get_storage(dial_api_client)


@pytest.fixture
def attachment_link(request_context):
    return AttachmentLink.from_link(
        request_context,
        "files/6iTkeGUs2CvUehhYLmMYXB/folder%201/file-example_PDF%20500_kB.pdf",
    )


class MockStage(Stage):
    def __init__(self, queue, index, last_stage_index, name=None):
        super().__init__(queue, index, last_stage_index, name)
        self.content = []

    def append_content(self, content):
        self.content.append(content)


@pytest.mark.asyncio
@patch(
    "aidial_rag.document_loaders.download_attachment", new_callable=AsyncMock
)
async def test_attachment_test(mock_fetch, request_context, attachment_link):
    mock_fetch.return_value = "application/pdf", b"This is a test byte array."
    absolute_url = attachment_link.absolute_url
    headers = request_context.get_file_access_headers(absolute_url)

    filename, _content_type, bytes_value = await load_attachment(
        attachment_link, headers
    )

    assert filename == "folder 1/file-example_PDF 500_kB.pdf"
    assert len(bytes_value) == 26  # Assuming you expect the length to be 50
    assert bytes_value == b"This is a test byte array."


@pytest.mark.asyncio
@patch("aidial_rag.documents.check_document_access", new_callable=AsyncMock)
@patch(
    "aidial_rag.document_loaders.download_attachment", new_callable=AsyncMock
)
@patch("aidial_sdk.chat_completion.Choice.create_stage")
async def test_load_document_success(
    mock_create_stage,
    mock_fetch,
    mock_check_document_access,
    request_context,
    dial_api_client,
    index_storage,
    attachment_link,
):
    mock_check_document_access.return_value = None
    mock_fetch.return_value = "text/plain", b"This is a test byte array."

    mock_create_stage.side_effect = lambda name=None: MockStage(
        MagicMock(), 0, 0, name
    )

    indexing_task = IndexingTask(
        attachment_link=attachment_link,
        index_url=link_to_index_url(attachment_link, dial_api_client.bucket_id),
    )

    # Download and store
    doc_record = await load_document(
        request_context,
        indexing_task,
        index_storage,
        ColpaliModelResource(None, request_config.indexing.colpali_index),
        config=request_config,
    )
    assert isinstance(doc_record, DocumentRecord)
    assert doc_record.document_bytes == b"This is a test byte array."

    index_settings = request_config.indexing.collect_fields_that_rebuild_index()

    # Read stored value
    doc = await index_storage.load(indexing_task, index_settings)
    assert isinstance(doc, DocumentRecord)
    assert doc.document_bytes == b"This is a test byte array."
    assert len(doc.chunks) == 1


@pytest.mark.asyncio
@patch("aidial_rag.documents.check_document_access", new_callable=AsyncMock)
@patch(
    "aidial_rag.document_loaders.download_attachment", new_callable=AsyncMock
)
@patch("aidial_sdk.chat_completion.Choice.create_stage")
async def test_load_document_invalid_document(
    mock_create_stage,
    mock_fetch,
    mock_check_document_access,
    request_context,
    dial_api_client,
    index_storage,
    attachment_link,
):
    mock_check_document_access.return_value = None
    mock_fetch.return_value = None, None

    mock_create_stage.side_effect = lambda name=None: MockStage(
        MagicMock(), 0, 0, name
    )

    dial_api_client = MockDialApiClient()
    index_url = link_to_index_url(attachment_link, dial_api_client.bucket_id)

    with pytest.raises(DocumentProcessingError) as exc_info:
        await load_document(
            request_context,
            IndexingTask(
                attachment_link=attachment_link,
                index_url=index_url,
            ),
            index_storage,
            ColpaliModelResource(None, request_config.indexing.colpali_index),
            config=request_config,
        )
    assert isinstance(exc_info.value.__cause__, InvalidDocumentError)
