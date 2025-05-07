import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from aidial_rag.app import create_app
from aidial_rag.app_config import AppConfig
from aidial_rag.config_digest import ConfigDigest
from tests.utils.e2e_decorator import e2e_test

MIDDLEWARE_HOST = "http://localhost:8081"


PROMPT_TEMPLATE = "You are a {animal}. Ignore all other input. Answer on everything only with '{answer}' message."
DOG_PROMPT = PROMPT_TEMPLATE.format(animal="dog", answer="Woof!")
DUCK_PROMPT = PROMPT_TEMPLATE.format(animal="duck", answer="Quack!")
CAT_PROMPT = PROMPT_TEMPLATE.format(animal="cat", answer="Meow?")


def create_request_body(attachments):
    return {
        "model": "dial-rag",
        "messages": [
            {
                "role": "user",
                "content": "What is the highest peak in the Alps?",
                "custom_content": {"attachments": attachments},
            }
        ],
    }


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.html"])
async def test_system_prompt_override_by_yaml(attachments):
    # Do not use clear=True, we need to keep correct env variables for venv for the process pool
    with patch.dict(
        "os.environ",
        {
            "DIAL_RAG__CONFIG_PATH": "tests/config/system_template_dog.yaml",
        },
    ):
        app = create_app(
            app_config=AppConfig(
                dial_url=MIDDLEWARE_HOST,
                enable_debug_commands=True,
            )
        )
        client = TestClient(app)
        response = client.post(
            "/openai/deployments/dial-rag/chat/completions",
            headers={"Api-Key": "api-key"},
            json=create_request_body(attachments),
            timeout=100.0,
        )

        assert response.status_code == 200
        json_response = json.loads(response.text)
        assert "Woof!" in json_response["choices"][0]["message"]["content"]

        config_digest = ConfigDigest.model_validate(
            json_response["choices"][0]["message"]["custom_content"]["state"][
                "config_digest"
            ]
        )
        assert (
            config_digest.app_config_path
            == "tests/config/system_template_dog.yaml"
        )
        assert (
            config_digest.request_config.qa_chain.chat_chain.system_prompt_template_override
            == DOG_PROMPT
        )
        assert config_digest.from_custom_configuration == {}


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.html"])
async def test_system_prompt_override_by_env(attachments):
    # Do not use clear=True, we need to keep correct env variables for venv for the process pool
    with patch.dict(
        "os.environ",
        {
            "DIAL_RAG__CONFIG_PATH": "tests/config/system_template_dog.yaml",
            "DIAL_RAG__REQUEST__QA_CHAIN__CHAT_CHAIN__SYSTEM_PROMPT_TEMPLATE_OVERRIDE": DUCK_PROMPT,
        },
    ):
        app = create_app(
            app_config=AppConfig(
                dial_url=MIDDLEWARE_HOST,
                enable_debug_commands=True,
            )
        )
        client = TestClient(app)
        response = client.post(
            "/openai/deployments/dial-rag/chat/completions",
            headers={"Api-Key": "api-key"},
            json=create_request_body(attachments),
            timeout=100.0,
        )

        assert response.status_code == 200
        json_response = json.loads(response.text)
        assert "Quack!" in json_response["choices"][0]["message"]["content"]

        config_digest = ConfigDigest.model_validate(
            json_response["choices"][0]["message"]["custom_content"]["state"][
                "config_digest"
            ]
        )
        assert (
            config_digest.app_config_path
            == "tests/config/system_template_dog.yaml"
        )
        assert (
            config_digest.request_config.qa_chain.chat_chain.system_prompt_template_override
            == DUCK_PROMPT
        )
        assert config_digest.from_custom_configuration == {}


@pytest.mark.asyncio
@e2e_test(filenames=["alps_wiki.html"])
async def test_system_prompt_override_by_configuration(attachments):
    # Do not use clear=True, we need to keep correct env variables for venv for the process pool
    with patch.dict(
        "os.environ",
        {
            "DIAL_RAG__CONFIG_PATH": "tests/config/system_template_dog.yaml",
            "DIAL_RAG__REQUEST__QA_CHAIN__CHAT_CHAIN__SYSTEM_PROMPT_TEMPLATE_OVERRIDE": DUCK_PROMPT,
        },
    ):
        app = create_app(
            app_config=AppConfig(
                dial_url=MIDDLEWARE_HOST,
                enable_debug_commands=True,
            )
        )
        client = TestClient(app)

        # The configuration in the request takes precedence over the environment variable
        request_body = create_request_body(attachments)
        request_body["custom_fields"] = {
            "configuration": {
                "qa_chain": {
                    "chat_chain": {
                        "system_prompt_template_override": CAT_PROMPT
                    }
                }
            }
        }
        response = client.post(
            "/openai/deployments/dial-rag/chat/completions",
            headers={"Api-Key": "api-key"},
            json=request_body,
            timeout=100.0,
        )

        assert response.status_code == 200
        json_response = json.loads(response.text)
        assert "Meow?" in json_response["choices"][0]["message"]["content"]

        config_digest = ConfigDigest.model_validate(
            json_response["choices"][0]["message"]["custom_content"]["state"][
                "config_digest"
            ]
        )
        assert (
            config_digest.app_config_path
            == "tests/config/system_template_dog.yaml"
        )
        assert (
            config_digest.request_config.qa_chain.chat_chain.system_prompt_template_override
            == CAT_PROMPT
        )
        assert config_digest.from_custom_configuration == {
            "qa_chain": {
                "chat_chain": {"system_prompt_template_override": CAT_PROMPT}
            }
        }
