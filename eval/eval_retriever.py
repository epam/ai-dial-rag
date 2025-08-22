import asyncio
import logging
import os
import sys
from operator import itemgetter
from typing import Dict, List, Tuple

# Onnx 1.16.2 and 1.17.0 have DLL initialization issues for Windows
#  https://github.com/onnx/onnx/issues/6267
# The onnx module should be imported before any unstructured_inference and pandas imports to avoid the issue
import onnx  # noqa: F401
import pandas as pd
from aidial_rag.configuration_endpoint import RequestConfig
from aidial_rag_eval.evaluate import evaluate
from langchain.schema import BaseRetriever
from langchain.schema.runnable import RunnablePassthrough
from pydantic import SecretStr

from aidial_rag.app_config import IndexingConfig
from aidial_rag.attachment_link import AttachmentLink
from aidial_rag.dial_config import DialConfig
from aidial_rag.document_record import DocumentRecord
from aidial_rag.documents import load_document_impl
from aidial_rag.resources.dial_limited_resources import DialLimitedResources
from aidial_rag.retrieval_chain import create_retriever
from aidial_rag.retrievers.colpali_retriever.colpali_model_resource import (
    ColpaliModelResource,
)
from aidial_rag.retrievers_postprocess import get_text_chunks
from tests.utils.local_http_server import start_local_server
from tests.utils.user_limits_mock import user_limits_mock

DATA_DIR = "eval/data"
OUT_DIR = "eval/out"
PORT = 5007


def metrics_to_list(
    metrics_dict: Dict[str, float|int]
) -> List[Tuple[str, float|int]]:
    return [
        (k.replace(" ", "_").lower(), v)
        for k, v in metrics_dict.items()
    ]


async def prepare_doc_records(
    document_link: str,
    request_config: RequestConfig,
):
    attachment_link = AttachmentLink(
        dial_link=document_link,
        absolute_url=document_link,
        display_name=document_link.split("/")[-1],
    )

    doc_record = await load_document_impl(
        dial_config=DialConfig(dial_url="-", api_key=SecretStr("-")),
        dial_limited_resources=DialLimitedResources(user_limits_mock()),
        attachment_link=attachment_link,
        stage_stream=sys.stderr,
        index_settings=request_config.indexing.collect_fields_that_rebuild_index(),
        config=request_config,
    )
    return [doc_record]


def prepare_retriever(
    doc_records: List[DocumentRecord],
    indexing_config: IndexingConfig,
):
    retriever = create_retriever(
        dial_config=DialConfig(dial_url="-", api_key=SecretStr("-")),
        document_records=doc_records,
        colpali_model_resource=None,
        indexing_config=indexing_config,
    )

    retriever_chain = RunnablePassthrough().assign(
        found_items=(itemgetter("query") | retriever)
    ) | get_text_chunks

    return retriever_chain


async def generate_retrieval_results(
    retriever: BaseRetriever,
    doc_records: List[DocumentRecord],
    ground_truth_path: str,
    output_path: str
):
    logging.info("Read questions...")
    ground_truth_df = pd.read_parquet(ground_truth_path)
    logging.info(f"Questions: {len(ground_truth_df)}")

    logging.info("Run retrieval...")
    inputs = [
        {"query": q, "doc_records": doc_records}
        for q in ground_truth_df["question"].to_list()
    ]
    contexts_docs = await retriever.abatch(inputs)
    context = [[d.page_content for d in docs] for docs in contexts_docs]

    answers_df = pd.DataFrame({
        "question": ground_truth_df["question"],
        "documents": ground_truth_df["documents"],
        "context": context,
        "answer": [""] * len(ground_truth_df),  # skip answer for retrieval evaluation
    })
    answers_df.to_parquet(output_path)


async def run_eval():
    DATA_DIR = "eval/data"
    OUT_DIR = "eval/out"

    request_config = RequestConfig(
        indexing=IndexingConfig(
            multimodal_index=None,
            description_index=None,
        ),
    )

    logging.info("Prepare retriever...")
    doc_records = await prepare_doc_records(
        f"http://localhost:{PORT}/alps_wiki.pdf",
        request_config,
    )
    retriever = prepare_retriever(
        doc_records,
        request_config.indexing,
    )

    ground_truth_path = os.path.join(DATA_DIR, "alps_ground_truth_mixtral_v2.parquet")

    os.makedirs(OUT_DIR, exist_ok=True)
    answers_path = os.path.join(OUT_DIR, "alps_answers.parquet")
    await generate_retrieval_results(retriever, doc_records, ground_truth_path, answers_path)

    logging.info("Evaluate...")
    eval_path = os.path.join(OUT_DIR, "alps_eval_result.parquet")
    eval_result = evaluate(ground_truth_path, answers_path, eval_path)
    logging.info(f"Metrics: {eval_result.metadata.metrics}")
    logging.info(f"Statistics: {eval_result.metadata.statistics}")

    metrics_list = metrics_to_list(eval_result.metadata.metrics) + metrics_to_list(eval_result.metadata.statistics)
    print(metrics_list)
    metrics_path = os.path.join(OUT_DIR, "alps_metrics.txt")
    with open(metrics_path, "w") as f:
        f.writelines([f"{k} {v}\n" for k, v in metrics_list])


if __name__ == "__main__":
    with start_local_server(data_dir=DATA_DIR, port=PORT) as server:
        asyncio.run(run_eval())
