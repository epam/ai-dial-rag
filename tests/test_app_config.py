from pathlib import Path
from unittest.mock import patch

import pytest

from aidial_rag.app_config import AppConfig
from aidial_rag.base_config import update_config_field


def test_multimodal_index_settings():
    config = AppConfig(config_path="config/gcp_embedding.yaml")

    index_settings = config.request.indexing.collect_fields_that_rebuild_index()
    assert index_settings.indexes == {
        "download": {},
        "multimodal_index": {
            "embeddings_model": "multimodalembedding@001",
        },
        "parser": {
            "unstructured_chunk_size": 1000,
        },
    }


def test_description_index_settings():
    config = AppConfig(config_path="config/azure_description.yaml")

    index_settings = config.request.indexing.collect_fields_that_rebuild_index()
    assert index_settings.indexes == {
        "download": {},
        "description_index": {},
        "parser": {
            "unstructured_chunk_size": 1000,
        },
    }


@pytest.mark.parametrize(
    "config_file",
    list(Path("config").glob("*.yaml")),
    ids=str,
)
def test_config_profiles(config_file):
    assert config_file.exists(), (
        f"Configuration file {config_file} does not exist"
    )

    config = AppConfig(config_path=config_file)
    assert config, f"Failed to load configuration from {config_file}"
    assert config.model_json_schema(), (
        f"Schema generation failed for {config_file}"
    )
    assert config.request.indexing.collect_fields_that_rebuild_index(), (
        f"Index settings generation failed for {config_file}"
    )


def test_config_update():
    config = AppConfig(config_path="config/azure_description.yaml")
    assert (
        config.request.qa_chain.chat_chain.llm.deployment_name
        == "gpt-4o-2024-05-13"
    )

    request_updated_config = update_config_field(
        config.request,
        "qa_chain.chat_chain.llm.deployment_name",
        "gpt-4o-updated",
    )
    assert (
        request_updated_config.qa_chain.chat_chain.llm.deployment_name
        == "gpt-4o-updated"
    )


def test_env_override_fields():
    with patch.dict(
        "os.environ",
        {
            "DIAL_URL": "http://custom-url.example",
            "ENABLE_DEBUG_COMMANDS": "true",
            "DIAL_RAG__REQUEST__USE_PROFILER": "true",
            "DIAL_RAG__REQUEST__QA_CHAIN__CHAT_CHAIN__LLM__DEPLOYMENT_NAME": "gpt-4-custom",
            "DIAL_RAG__REQUEST__QA_CHAIN__CHAT_CHAIN__LLM__MAX_PROMPT_TOKENS": "12000",
        },
        clear=True,
    ):
        config = AppConfig()
        assert config.dial_url == "http://custom-url.example"
        assert config.enable_debug_commands is True
        assert config.request.use_profiler is True
        assert (
            config.request.qa_chain.chat_chain.llm.deployment_name
            == "gpt-4-custom"
        )
        assert config.request.qa_chain.chat_chain.llm.max_prompt_tokens == 12000


def test_env_override_and_yaml_config():
    with patch.dict(
        "os.environ",
        {
            "DIAL_RAG__REQUEST__QA_CHAIN__CHAT_CHAIN__LLM__DEPLOYMENT_NAME": "gpt-4-custom"
        },
        clear=True,
    ):
        config = AppConfig(config_path="config/gcp_embedding.yaml")
        assert (
            config.request.qa_chain.chat_chain.llm.deployment_name
            == "gpt-4-custom"
        )
        # Confirm other settings still load from YAML
        assert config.request.indexing.description_index is None
        assert config.request.indexing.multimodal_index is not None
        assert (
            config.request.indexing.multimodal_index.embeddings_model
            == "multimodalembedding@001"
        )


def test_env_override_description_index_fields():
    with patch.dict(
        "os.environ",
        {
            "DIAL_RAG__REQUEST__INDEXING__DESCRIPTION_INDEX__LLM__DEPLOYMENT_NAME": "gpt-4-custom-desc",
            "DIAL_RAG__REQUEST__INDEXING__DESCRIPTION_INDEX__ESTIMATED_TASK_TOKENS": "5000",
        },
        clear=True,
    ):
        config = AppConfig()
        assert config.request.indexing.description_index is not None
        assert (
            config.request.indexing.description_index.llm.deployment_name
            == "gpt-4-custom-desc"
        )
        assert (
            config.request.indexing.description_index.estimated_task_tokens
            == 5000
        )


def test_env_override_full_description_index_config():
    complex_json = '{"llm": {"deployment_name": "gpt-4-complex", "max_retries": 5, "max_prompt_tokens": 2000}, "estimated_task_tokens": 6000}'
    with patch.dict(
        "os.environ",
        {"DIAL_RAG__REQUEST__INDEXING__DESCRIPTION_INDEX": complex_json},
        clear=True,
    ):
        config = AppConfig()
        assert config.request.indexing.description_index is not None
        assert (
            config.request.indexing.description_index.llm.deployment_name
            == "gpt-4-complex"
        )
        assert config.request.indexing.description_index.llm.max_retries == 5
        assert (
            config.request.indexing.description_index.llm.max_prompt_tokens
            == 2000
        )
        assert (
            config.request.indexing.description_index.estimated_task_tokens
            == 6000
        )


def test_env_override_index_type():
    with patch.dict(
        "os.environ",
        {
            "DIAL_RAG__REQUEST__INDEXING__DESCRIPTION_INDEX": "null",  # Disable description index
            "DIAL_RAG__REQUEST__INDEXING__MULTIMODAL_INDEX__EMBEDDINGS_MODEL": "env-multimodal-model",
        },
        clear=True,
    ):
        config = AppConfig(config_path="config/azure_description.yaml")
        assert config.request.indexing.description_index is None
        assert config.request.indexing.multimodal_index is not None
        assert (
            config.request.indexing.multimodal_index.embeddings_model
            == "env-multimodal-model"
        )
        assert (
            config.request.indexing.multimodal_index.metric
            == "sqeuclidean_dist"
        )


def test_env_system_prompt_override():
    with patch.dict(
        "os.environ",
        {
            "DIAL_RAG__REQUEST__QA_CHAIN__CHAT_CHAIN__SYSTEM_PROMPT_TEMPLATE_OVERRIDE": "You are a duck."
        },
        clear=True,
    ):
        config = AppConfig(config_path="config/azure_description.yaml")
        assert (
            config.request.qa_chain.chat_chain.system_prompt_template_override
            == "You are a duck."
        )
