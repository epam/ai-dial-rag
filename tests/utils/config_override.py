from unittest.mock import patch

import pytest


@pytest.fixture()
def description_index_retries_override():
    """In normal operation, the description index will use infinite retries and rely on
    the timeout to stop the process. This fixture overrides the configuration to reduce
    the wait time in case of a test failure, since tests use cached responses anyway.
    """
    with patch.dict(
        "os.environ",
        {
            "DIAL_RAG__REQUEST__INDEXING__DESCRIPTION_INDEX__LLM__MAX_RETRIES": "0",
            "DIAL_RAG__REQUEST__INDEXING__DESCRIPTION_INDEX__MIN_TIME_LIMIT_SEC": "30",
        },
    ):
        yield
