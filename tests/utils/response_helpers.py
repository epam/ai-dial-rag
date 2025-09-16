import json
import re
from typing import Any, Dict, List

from aidial_rag.index_mime_type import INDEX_MIME_TYPE
from aidial_rag.indexing_api import IndexingResponse
from aidial_rag.retrieval_api import RetrievalResponse


def get_stage_names(response_json: Dict[str, Any]) -> List[str]:
    stages = response_json["choices"][0]["message"]["custom_content"]["stages"]
    return [
        re.sub(
            r"\s*\[.*\]$", "", stage["name"]
        ).strip()  # cut [0.03s] at the end of the stage name
        for stage in stages
    ]


def get_attachments(json_response):
    attachments = json_response["choices"][0]["message"]["custom_content"][
        "attachments"
    ]

    return attachments


def get_index_attachments(attachments):
    index_attachments_result = [
        attachment
        for attachment in attachments
        if attachment["type"] == INDEX_MIME_TYPE
    ]

    return index_attachments_result


def get_indexing_result_json(attachments):
    indexing_result_attachment = next(
        attachment
        for attachment in attachments
        if attachment["type"] == IndexingResponse.CONTENT_TYPE
    )

    indexing_result_json = json.loads(indexing_result_attachment["data"])
    return indexing_result_json


def get_retrieval_response_json(attachments):
    retrieval_response_attachment = next(
        attachment
        for attachment in attachments
        if attachment["type"] == RetrievalResponse.CONTENT_TYPE
    )

    retrieval_response_json = json.loads(retrieval_response_attachment["data"])
    return retrieval_response_json
