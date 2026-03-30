import asyncio
from unittest.mock import MagicMock, patch

import pytest
from aidial_sdk.chat_completion import Choice
from aioresponses import aioresponses
from docarray import DocList
from pydantic import SecretStr

from aidial_rag.configuration_endpoint import Configuration
from aidial_rag.dial_api_client import DialApiClient
from aidial_rag.dial_config import DialConfig
from aidial_rag.document_metadata import parse_last_modified
from aidial_rag.document_record import (
    FORMAT_VERSION,
    Chunk,
    DocumentRecord,
    ModificationMetadata,
    serialize_document_record,
)
from aidial_rag.document_record_migration import MIN_FORMAT_VERSION
from aidial_rag.documents import load_document
from aidial_rag.embeddings.local_embeddings import (
    create_local_bge_embeddings_model,
)
from aidial_rag.errors import (
    DocumentProcessingError,
    IndexMissingError,
)
from aidial_rag.index_storage import IndexStorage
from aidial_rag.indexing_config import IndexingConfig
from aidial_rag.indexing_task import AttachmentLink, IndexingTask
from aidial_rag.request_context import RequestContext
from aidial_rag.resources.dial_limited_resources import DialLimitedResources
from aidial_rag.retrievers.multimodal_retriever import MultimodalIndexConfig
from tests.utils.user_limits_mock import user_limits_mock


@pytest.fixture
def request_context():
    choice = Choice(queue=asyncio.Queue(), choice_index=0)
    choice.open()
    return RequestContext(
        dial_config=DialConfig(
            dial_url="http://localhost:8080",
            api_key=SecretStr("api-key"),
        ),
        choice=choice,
        dial_limited_resources=DialLimitedResources(user_limits_mock()),
    )


@pytest.fixture
def dial_api_client():
    client = MagicMock(spec=DialApiClient)
    client.bucket_id = "dial_rag_bucket"
    return client


@pytest.fixture
def index_storage():
    return MagicMock(spec=IndexStorage)


@pytest.fixture
def aio_mock():
    with aioresponses() as m:
        yield m


def _make_doc_record(
    config: Configuration,
    modification_metadata: ModificationMetadata | None = None,
):
    return DocumentRecord(
        format_version=FORMAT_VERSION,
        index_settings=config.indexing.collect_fields_that_rebuild_index(),
        chunks=DocList[Chunk](),
        text_index=None,
        embeddings_index=None,
        mime_type="application/pdf",
        document_bytes=b"",
        modification_metadata=modification_metadata or ModificationMetadata(),
    )


def _make_dial_metadata_response(url: str, etag: str):
    return {
        "name": url.split("/")[-1],
        "author": "test_user",
        "parentPath": "",
        "bucket": "user_bucket",
        "url": url,
        "nodeType": "ITEM",
        "resourceType": "FILE",
        "createdAt": 1758648364,
        "updatedAt": 1758648364,
        "contentLength": 12345,
        "contentType": "application/pdf",
        "etag": etag,
    }


def _mock_dial_metadata_request(
    aio_mock: aioresponses, url: str, etag: str, status: int = 200
):
    aio_mock.get(
        f"http://localhost:8080/v1/metadata/{url}",
        payload=_make_dial_metadata_response(
            url=url,
            etag=etag,
        ),
        status=status,
    )


def _mock_external_head_request(
    aio_mock: aioresponses,
    url: str,
    etag: str | None = None,
    last_modified: str | None = None,
    status: int = 200,
):
    headers = {
        "Content-Length": "518216",
        "Content-Type": "text/html; charset=UTF-8",
    }
    if etag:
        headers["ETag"] = etag
    if last_modified:
        headers["Last-Modified"] = last_modified

    aio_mock.head(
        url,
        headers=headers,
        status=status,
    )


def _make_indexing_task(
    request_context: RequestContext,
    link: str,
    index_url: str = "/dial-rag-index/65cbfece/02adda18/06edc015/cafb091f/154ec3b7/141eec9a/2270b563/b0d827d8/index.bin",
):
    return IndexingTask(
        attachment_link=AttachmentLink.from_link(request_context, link=link),
        index_url=index_url,
    )


@pytest.mark.asyncio
async def test_new_dial_file(
    request_context, dial_api_client, index_storage, aio_mock
):
    config = Configuration()

    task = _make_indexing_task(
        request_context, link="files/user_bucket/test.pdf"
    )

    index_storage.load.side_effect = IndexMissingError()

    _mock_dial_metadata_request(
        aio_mock, url="files/user_bucket/test.pdf", etag='"abc123"'
    )

    with patch(
        "aidial_rag.documents.load_document_impl"
    ) as load_document_impl_mock:
        load_document_impl_mock.return_value = _make_doc_record(
            config, ModificationMetadata(etag='"abc123"')
        )

        result_doc_record = await load_document(
            request_context,
            task,
            index_storage,
            dial_api_client,
            create_local_bge_embeddings_model(),
            config,
        )

        index_storage.load.assert_called_once()
        load_document_impl_mock.assert_called_once()

    assert result_doc_record is not None
    assert result_doc_record.modification_metadata.etag == '"abc123"'
    assert result_doc_record.modification_metadata.last_modified is None


@pytest.mark.asyncio
async def test_dial_file_not_modified(
    request_context, dial_api_client, index_storage, aio_mock
):
    config = Configuration()

    task = _make_indexing_task(
        request_context, link="files/user_bucket/test.pdf"
    )

    index_storage.load.return_value = serialize_document_record(
        _make_doc_record(
            config, modification_metadata=ModificationMetadata(etag='"abc123"')
        )
    )
    _mock_dial_metadata_request(
        aio_mock, url="files/user_bucket/test.pdf", etag='"abc123"'
    )

    with patch(
        "aidial_rag.documents.load_document_impl"
    ) as load_document_impl_mock:
        result_doc_record = await load_document(
            request_context,
            task,
            index_storage,
            dial_api_client,
            create_local_bge_embeddings_model(),
            config,
        )

        index_storage.load.assert_called_once()
        load_document_impl_mock.assert_not_called()

    assert result_doc_record is not None
    assert result_doc_record.modification_metadata.etag == '"abc123"'
    assert result_doc_record.modification_metadata.last_modified is None


@pytest.mark.asyncio
async def test_dial_file_modified(
    request_context, dial_api_client, index_storage, aio_mock
):
    config = Configuration()

    task = _make_indexing_task(
        request_context, link="files/user_bucket/test.pdf"
    )

    index_storage.load.return_value = serialize_document_record(
        _make_doc_record(
            config, modification_metadata=ModificationMetadata(etag='"abc123"')
        )
    )
    _mock_dial_metadata_request(
        aio_mock, url="files/user_bucket/test.pdf", etag='"def456"'
    )

    with patch(
        "aidial_rag.documents.load_document_impl"
    ) as load_document_impl_mock:
        # The file could be modified again after the metadata request
        load_document_impl_mock.return_value = _make_doc_record(
            config, ModificationMetadata(etag='"ghi789"')
        )

        result_doc_record = await load_document(
            request_context,
            task,
            index_storage,
            dial_api_client,
            create_local_bge_embeddings_model(),
            config,
        )

        index_storage.load.assert_called_once()
        load_document_impl_mock.assert_called_once()

    assert result_doc_record is not None
    # The etag should be from the actual file load, not from the metadata request
    assert result_doc_record.modification_metadata.etag == '"ghi789"'
    assert result_doc_record.modification_metadata.last_modified is None


@pytest.mark.asyncio
async def test_dial_file_no_modification_metadata(
    request_context, dial_api_client, index_storage, aio_mock
):
    config = Configuration()

    task = _make_indexing_task(
        request_context, link="files/user_bucket/test.pdf"
    )

    index_storage.load.return_value = serialize_document_record(
        _make_doc_record(
            config,
            modification_metadata=None,  # No modification metadata in the old index
        )
    )
    _mock_dial_metadata_request(
        aio_mock, url="files/user_bucket/test.pdf", etag='"abc123"'
    )

    with patch(
        "aidial_rag.documents.load_document_impl"
    ) as load_document_impl_mock:
        # The file could be modified again after the metadata request
        load_document_impl_mock.return_value = _make_doc_record(
            config, ModificationMetadata(etag='"def456"')
        )

        result_doc_record = await load_document(
            request_context,
            task,
            index_storage,
            dial_api_client,
            create_local_bge_embeddings_model(),
            config,
        )

        index_storage.load.assert_called_once()
        load_document_impl_mock.assert_called_once()

    assert result_doc_record is not None
    assert result_doc_record.modification_metadata.etag == '"def456"'
    assert result_doc_record.modification_metadata.last_modified is None


@pytest.mark.asyncio
async def test_dial_file_modified_with_indexing_disabled(
    request_context, dial_api_client, index_storage, aio_mock
):
    config = Configuration()
    config.request.allow_indexing = False

    task = _make_indexing_task(
        request_context, link="files/user_bucket/test.pdf"
    )

    index_storage.load.return_value = serialize_document_record(
        _make_doc_record(
            config, modification_metadata=ModificationMetadata(etag='"abc123"')
        )
    )
    _mock_dial_metadata_request(
        aio_mock, url="files/user_bucket/test.pdf", etag='"def456"'
    )

    with patch(
        "aidial_rag.documents.load_document_impl"
    ) as load_document_impl_mock:
        with pytest.raises(DocumentProcessingError) as e:
            await load_document(
                request_context,
                task,
                index_storage,
                dial_api_client,
                create_local_bge_embeddings_model(),
                config,
            )

        index_storage.load.assert_called_once()
        load_document_impl_mock.assert_not_called()
        assert "Error on processing document: Document was modified" == str(
            e.value
        )


@pytest.mark.asyncio
async def test_dial_file_force_indexing(
    request_context, dial_api_client, index_storage, aio_mock
):
    config = Configuration()
    config.request.force_indexing = True

    task = _make_indexing_task(
        request_context, link="files/user_bucket/test.pdf"
    )

    index_storage.load.return_value = serialize_document_record(
        _make_doc_record(
            config, modification_metadata=ModificationMetadata(etag='"abc123"')
        )
    )
    _mock_dial_metadata_request(
        aio_mock, url="files/user_bucket/test.pdf", etag='"abc123"'
    )

    with patch(
        "aidial_rag.documents.load_document_impl"
    ) as load_document_impl_mock:
        # The file is the same, but force_indexing is True, so the index needs to be rebuilt
        load_document_impl_mock.return_value = _make_doc_record(
            config, ModificationMetadata(etag='"abc123"')
        )

        result_doc_record = await load_document(
            request_context,
            task,
            index_storage,
            dial_api_client,
            create_local_bge_embeddings_model(),
            config,
        )

        # Do not check index_storage.load, because it depends on the implementation details
        load_document_impl_mock.assert_called_once()

    assert result_doc_record is not None
    assert result_doc_record.modification_metadata.etag == '"abc123"'
    assert result_doc_record.modification_metadata.last_modified is None


@pytest.mark.asyncio
async def test_dial_file_ignore_file_modification(
    request_context, dial_api_client, index_storage, aio_mock
):
    config = Configuration()
    config.request.ignore_file_modification = True

    task = _make_indexing_task(
        request_context,
        link="files/user_bucket/new_folder/test.pdf",
        index_url="files/user_bucket/new_folder/test_pdf_index.bin",
    )

    index_storage.load.return_value = serialize_document_record(
        _make_doc_record(
            config, modification_metadata=ModificationMetadata(etag='"abc123"')
        )
    )
    _mock_dial_metadata_request(
        aio_mock, url="files/user_bucket/new_folder/test.pdf", etag='"def456"'
    )

    with patch(
        "aidial_rag.documents.load_document_impl"
    ) as load_document_impl_mock:
        result_doc_record = await load_document(
            request_context,
            task,
            index_storage,
            dial_api_client,
            create_local_bge_embeddings_model(),
            config,
        )

        index_storage.load.assert_called_once()
        load_document_impl_mock.assert_not_called()

    assert result_doc_record is not None
    # Old index is used, even if the file was modified
    assert result_doc_record.modification_metadata.etag == '"abc123"'
    assert result_doc_record.modification_metadata.last_modified is None


@pytest.mark.asyncio
async def test_dial_file_forbidden(
    request_context, dial_api_client, index_storage, aio_mock
):
    config = Configuration()

    task = _make_indexing_task(
        request_context, link="files/user_bucket/test.pdf"
    )

    _mock_dial_metadata_request(
        aio_mock, url="files/user_bucket/test.pdf", etag='"abc123"', status=403
    )

    with patch(
        "aidial_rag.documents.load_document_impl"
    ) as load_document_impl_mock:
        with pytest.raises(DocumentProcessingError) as e:
            await load_document(
                request_context,
                task,
                index_storage,
                dial_api_client,
                create_local_bge_embeddings_model(),
                config,
            )

        index_storage.load.assert_not_called()
        load_document_impl_mock.assert_not_called()
        assert "Error on processing document: 403 Forbidden" == str(e.value)


@pytest.mark.asyncio
async def test_new_external_file(
    request_context, dial_api_client, index_storage, aio_mock
):
    config = Configuration()

    task = _make_indexing_task(
        request_context, link="https://en.wikipedia.org/wiki/Alps"
    )

    index_storage.load.side_effect = IndexMissingError()

    _mock_external_head_request(
        aio_mock,
        url="https://en.wikipedia.org/wiki/Alps",
        last_modified="Sun, 28 Sep 2025 06:12:48 GMT",
    )

    with patch(
        "aidial_rag.documents.load_document_impl"
    ) as load_document_impl_mock:
        load_document_impl_mock.return_value = _make_doc_record(
            config,
            ModificationMetadata(
                last_modified=1759039968,
            ),
        )

        result_doc_record = await load_document(
            request_context,
            task,
            index_storage,
            dial_api_client,
            create_local_bge_embeddings_model(),
            config,
        )

        index_storage.load.assert_called_once()
        load_document_impl_mock.assert_called_once()

    assert result_doc_record is not None
    assert result_doc_record.modification_metadata.last_modified == 1759039968
    assert result_doc_record.modification_metadata.etag is None


@pytest.mark.parametrize(
    "index_etag, index_last_modified, actual_etag, actual_last_modified, expected_modified",
    [
        (
            '"abc123"',
            None,
            '"abc123"',
            None,
            False,
        ),
        (
            None,
            1759039968,
            None,
            "Sun, 28 Sep 2025 06:12:48 GMT",
            False,
        ),
        (
            '"abc123"',
            None,
            '"def456"',
            None,
            True,
        ),
        (
            None,
            1759030000,
            None,
            "Sun, 28 Sep 2025 06:12:48 GMT",
            True,
        ),
        (
            '"abc123"',
            1759039968,
            '"abc123"',
            "Sun, 28 Sep 2025 06:12:48 GMT",
            False,
        ),
        (
            '"abc123"',
            1759030000,
            '"def456"',
            "Sun, 28 Sep 2025 06:12:48 GMT",
            True,
        ),
    ],
)
@pytest.mark.asyncio
async def test_external_file_modified(
    index_etag,
    index_last_modified,
    actual_etag,
    actual_last_modified,
    expected_modified,
    request_context,
    dial_api_client,
    index_storage,
    aio_mock,
):
    config = Configuration()

    task = _make_indexing_task(
        request_context, link="https://en.wikipedia.org/wiki/Alps"
    )

    index_storage.load.return_value = serialize_document_record(
        _make_doc_record(
            config,
            modification_metadata=ModificationMetadata(
                last_modified=index_last_modified,
                etag=index_etag,
            ),
        )
    )

    _mock_external_head_request(
        aio_mock,
        url="https://en.wikipedia.org/wiki/Alps",
        last_modified=actual_last_modified,
        etag=actual_etag,
    )

    with patch(
        "aidial_rag.documents.load_document_impl"
    ) as load_document_impl_mock:
        if expected_modified:
            load_document_impl_mock.return_value = _make_doc_record(
                config,
                ModificationMetadata(
                    last_modified=parse_last_modified(actual_last_modified)
                    if actual_last_modified
                    else None,
                    etag=actual_etag,
                ),
            )

        result_doc_record = await load_document(
            request_context,
            task,
            index_storage,
            dial_api_client,
            create_local_bge_embeddings_model(),
            config,
        )

        index_storage.load.assert_called_once()
        if expected_modified:
            load_document_impl_mock.assert_called_once()
        else:
            load_document_impl_mock.assert_not_called()

    assert result_doc_record is not None


@pytest.mark.asyncio
async def test_incompatible_index_settings(
    request_context, dial_api_client, index_storage, aio_mock
):
    old_config = Configuration()
    assert old_config.indexing.description_index is not None
    assert old_config.indexing.multimodal_index is None

    # Assume that the deployed configuration was changed from description to multimodal
    new_config = Configuration(
        indexing=IndexingConfig(
            description_index=None,
            multimodal_index=MultimodalIndexConfig(
                embeddings_model="multimodalembedding@001",
            ),
        )
    )

    task = _make_indexing_task(
        request_context, link="files/user_bucket/test.pdf"
    )

    index_storage.load.return_value = serialize_document_record(
        DocumentRecord(
            format_version=FORMAT_VERSION,
            index_settings=old_config.indexing.collect_fields_that_rebuild_index(),
            chunks=DocList[Chunk](),
            text_index=None,
            embeddings_index=None,
            mime_type="application/pdf",
            document_bytes=b"",
            modification_metadata=ModificationMetadata(
                etag='"abc123"',
                last_modified=None,
            ),
        )
    )

    _mock_dial_metadata_request(
        aio_mock, url="files/user_bucket/test.pdf", etag='"abc123"'
    )

    with patch(
        "aidial_rag.documents.load_document_impl"
    ) as load_document_impl_mock:
        # The source document is the same, but the index settings are incompatible,
        # so the index needs to be rebuilt
        load_document_impl_mock.return_value = _make_doc_record(
            new_config,
            ModificationMetadata(
                etag='"abc123"',
            ),
        )

        result_doc_record = await load_document(
            request_context,
            task,
            index_storage,
            dial_api_client,
            create_local_bge_embeddings_model(),
            new_config,
        )

        index_storage.load.assert_called_once()
        load_document_impl_mock.assert_called_once()

    assert result_doc_record is not None
    assert result_doc_record.modification_metadata.etag == '"abc123"'
    assert result_doc_record.modification_metadata.last_modified is None

    expected_index_settings = (
        new_config.indexing.collect_fields_that_rebuild_index()
    )
    assert result_doc_record.index_settings == expected_index_settings


@pytest.mark.asyncio
async def test_old_index_format(
    request_context, dial_api_client, index_storage, aio_mock
):
    config = Configuration()

    task = _make_indexing_task(
        request_context, link="files/user_bucket/test.pdf"
    )

    index_storage.load.return_value = serialize_document_record(
        DocumentRecord(
            format_version=MIN_FORMAT_VERSION,
            index_settings=config.indexing.collect_fields_that_rebuild_index(),
            chunks=DocList[Chunk](),
            text_index=None,
            embeddings_index=None,
            mime_type="application/pdf",
            document_bytes=b"",
            modification_metadata=ModificationMetadata(),
        )
    )

    _mock_dial_metadata_request(
        aio_mock, url="files/user_bucket/test.pdf", etag='"abc123"'
    )

    with patch(
        "aidial_rag.documents.load_document_impl"
    ) as load_document_impl_mock:
        # The old index format does not have modification metadata,
        # so the document needs to be reloaded to update the index format
        load_document_impl_mock.return_value = _make_doc_record(
            config,
            ModificationMetadata(
                etag='"abc123"',
            ),
        )

        result_doc_record = await load_document(
            request_context,
            task,
            index_storage,
            dial_api_client,
            create_local_bge_embeddings_model(),
            config,
        )

        index_storage.load.assert_called_once()
        load_document_impl_mock.assert_called_once()

    assert result_doc_record is not None
    assert result_doc_record.modification_metadata.etag == '"abc123"'
    assert result_doc_record.modification_metadata.last_modified is None
    assert result_doc_record.format_version == FORMAT_VERSION
