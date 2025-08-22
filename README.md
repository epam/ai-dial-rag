# Dial RAG

## Overview

The Dial RAG answers user questions using information from the documents provided by user. It supports the following document formats: PDF, DOC/DOCX, PPT/PPTX, TXT and other plain text formats such as code files. Also, it supports PDF and JPEG, PNG and other image formats for the image understanding.

The Dial RAG implements several retrieval methods to find the relevant information:
* **Description retriever** - uses vision model to generate page images descriptions and perform search on them. Supports different vision models, like `gpt-4o-mini`, `gemini-1.5-flash-002` or `anthropic.claude-v3-haiku`.
* **Multimodal retriever** - uses multimodal embedding models for pages images search. Supports different multimodal models, like [`azure-ai-vision-embeddings`](https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/concept-image-retrieval), [Google `multimodalembedding@001`](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings) or [`amazon.titan-embed-image-v1`](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-multiemb-models.html)
* **Semantic retriever** - uses [text embedding model](https://huggingface.co/epam/bge-small-en) to find the relevant information in the documents.
* **Keyword retriever** - uses [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) algorithm to find the relevant information in the documents.

The Dial RAG is intended to be used in the [Dial](https://github.com/epam/ai-dial) environment. It uses the [Dial Core](https://github.com/epam/ai-dial-core) to access the LLMs and other services.

## Configuration

### Required environment variables

Following environment variables are required to set for the deployment configuration:

|Variable|Description|
|---|---|
|`DIAL_URL`| url to the dial core |
|`DIAL_RAG__INDEX_STORAGE__USE_DIAL_FILE_STORAGE`| set to **True** to store indexes in the Dial File Storage instead of the local file storage |

### Configuration files

The Dial RAG provides a set of configuration files with predefined settings for different environments. The configuration files are located in the `config` directory.
You can set the environment variable `DIAL_RAG__CONFIG_PATH` to point to the required configuration file depending on the Dial environment and available models.

The following configuration files are available in the `config` directory:
- `config/aws_description.yaml` - AWS environment with description retriever, which uses `Claude 3 Haiku` model for page images descriptions and `Claude 3.5 Sonnet` for the answer generation.
- `config/aws_embedding.yaml` - AWS environment with multimodal retriever, which uses `amazon.titan-embed-image-v1` model for page images embeddings and `Claude 3.5 Sonnet` for the answer generation.
- `config/azure_description.yaml` - Azure environment with description retriever, which uses `GPT-4o mini` model for page images descriptions and `GPT-4o` for the answer generation.
- `config/azure_embedding.yaml` - Azure environment with multimodal retriever, which uses `azure-ai-vision-embeddings` model for page images embeddings and `GPT-4o` for the answer generation.
- `config/gcp_description.yaml` - GCP environment with description retriever, which uses `Gemini 1.5 Flash` model for page images descriptions and `Gemini 1.5 Pro` for the answer generation.
- `config/gcp_embedding.yaml` - GCP environment with multimodal retriever, which uses Google `multimodalembedding@001` model for page images embeddings and `Gemini 1.5 Pro` for the answer generation.
- `config/azure_with_gcp_embedding.yaml` - mixed environment which assumes that you have and access to both Azure and GCP models in the Dial. It uses Google `multimodalembedding@001` model for page images embeddings and `GPT-4o` for the answer generation.

If you are running the Dial RAG in a different environment, you can create your own configuration file based on one of the provided files and set the `DIAL_RAG__CONFIG_PATH` environment variable to point to it. If you need a small change in the configuration (for example to change the model name), you can point the `DIAL_RAG__CONFIG_PATH` to the existing file and override the required settings using the environment variables. See the [Additional environment variables](#additional-environment-variables) section for the list of available settings.


### Logging configuration environment variables

|Variable|Default|Description|
|---|---|---|
|`LOG_LEVEL`| `INFO` | Log level for the application. |
|`LOG_LEVEL_OVERRIDE`| `{}` | Allows to override log level for specific modules. Example: `LOG_LEVEL_OVERRIDE='{"dial_rag": "DEBUG", "urllib3": "ERROR" }'`|


### Additional environment variables

Dial RAG has additional variables to tune its performance and behavior:

<!-- The following section of the file is automatically generated from the AppConfig. -->
<!-- Do NOT edit it manually. Use `nox -s update_docs` command for update. -->

<!-- generated-app-config-env-start -->
##### `DIAL_RAG__CONFIG_PATH`

*Optional*, default value: `.`

Path to the yaml configuration file.See config directory for examples.

##### `DIAL_URL`

*Optional*, default value: `http://dial-proxy.dial-proxy`

Url to the dial core.

##### `ENABLE_DEBUG_COMMANDS`

*Optional*, default value: `False`

Enables support of debug commands in the messages. Should be `false` for prod envs. It is set to `true` only for staging. See [Debug commands](README.md#debug-commands) for more details.

##### `DIAL_RAG__CPU_POOLS__INDEXING_CPU_POOL`

*Optional*, default value: `6`

Process pool for document parsing, image extraction and similar CPU-bound tasks. Is set to `max(1, CPU_COUNT - 2)` to leave some CPU cores for other tasks.

##### `DIAL_RAG__CPU_POOLS__INDEXING_EMBEDDINGS_POOL`

*Optional*, default value: `1`

Embedding process itself uses multiple cores. Should be `1`, unless you have a lot of cores and can explicitly see the underutilisation (i.e. you only have a very small documents in the requests).

##### `DIAL_RAG__CPU_POOLS__QUERY_EMBEDDINGS_POOL`

*Optional*, default value: `1`

Embedding process for the query. Should be `1`, unless you have a lot of cores.

##### `USE_DIAL_FILE_STORAGE`

*Optional*, default value: `False`

Set to `True` to store indexes in the Dial File Storage instead of in memory storage

##### `DIAL_RAG__INDEX_STORAGE__IN_MEMORY_CACHE_CAPACITY`

*Optional*, default value: `128MiB`

Used to cache the document indexes and avoid requesting Dial Core File API every time, if user makes several requests for the same document. Could be increased to reduce load on the Dial Core File API if we have a lot of concurrent users (requires corresponding increase of the pod memory). Could be integer for bytes, or a pydantic.ByteSize compatible string (e.g. 128MiB, 1GiB, 2.5GiB).

##### `DIAL_RAG__REQUEST__IGNORE_DOCUMENT_LOADING_ERRORS`

*Optional*, default value: `False`

Ignore errors during document loading. Used for Web RAG for the request with multiple documents.

##### `DIAL_RAG__REQUEST__USE_PROFILER`

*Optional*, default value: `False`

Use profiler to collect performance metrics for the request.

##### `DIAL_RAG__REQUEST__LOG_DOCUMENT_LINKS`

*Optional*, default value: `False`

Allows writing the links of the attached documents to the logs with log levels higher than DEBUG.

If enabled, Dial RAG will log the links to the documents for log messages with levels from INFO to CRITICAL where relevant. For example, an ERROR log message with an exception during document processing will contain the link to the document.

If disabled, only log messages with DEBUG level may contain the links to the documents, to avoid logging sensitive information. For example, the links to the documents will not be logged for the ERROR log messages with an exception during document processing.

##### `DIAL_RAG__REQUEST__DOWNLOAD__TIMEOUT_SECONDS`

*Optional*, default value: `30`

Timeout for the whole request. Includes connection establishment, sending the request, and receiving the response.

##### `DIAL_RAG__REQUEST__DOWNLOAD__CONNECT_TIMEOUT_SECONDS`

*Optional*, default value: `30`

Timeout for establishing a connection to the server.

##### `DIAL_RAG__REQUEST__CHECK_ACCESS__TIMEOUT_SECONDS`

*Optional*, default value: `30`

Timeout for the whole request. Includes connection establishment, sending the request, and receiving the response.

##### `DIAL_RAG__REQUEST__CHECK_ACCESS__CONNECT_TIMEOUT_SECONDS`

*Optional*, default value: `30`

Timeout for establishing a connection to the server.

##### `DIAL_RAG__REQUEST__INDEXING__PARSER__MAX_DOCUMENT_TEXT_SIZE`

*Optional*, default value: `5MiB`

Limits the size of the document the RAG will accept for processing. This limit is applied to the size of the text extracted from the document, not the size of the attached document itself. Could be integer for bytes, or a pydantic.ByteSize compatible string.

##### `DIAL_RAG__REQUEST__INDEXING__PARSER__UNSTRUCTURED_CHUNK_SIZE`

*Optional*, default value: `1000`

Sets the chunk size for unstructured document loader.

##### `DIAL_RAG__REQUEST__INDEXING__MULTIMODAL_INDEX`

*Optional*, default value: `None`

Enables MultimodalRetriever which uses multimodal embedding models for pages images search.

##### `DIAL_RAG__REQUEST__INDEXING__DESCRIPTION_INDEX`

*Optional*, default value: `llm=LlmConfig(deployment_name='gpt-4o-mini-2024-07-18', max_prompt_tokens=0, max_retries=1000000000) estimated_task_tokens=4000 time_limit_multiplier=1.5 min_time_limit_sec=300`

Enables DescriptionRetriever which uses vision model to generate page images descriptions and perform search on them.

##### `DIAL_RAG__REQUEST__QA_CHAIN__CHAT_CHAIN__LLM__DEPLOYMENT_NAME`

*Optional*, default value: `gpt-4o-2024-05-13`

Used to set the deployment name of the LLM used in the chain. Could be useful if the model deployments have non-standard names in the Dial Core configuration.

##### `DIAL_RAG__REQUEST__QA_CHAIN__CHAT_CHAIN__LLM__MAX_PROMPT_TOKENS`

*Optional*, default value: `0`

Sets `max_prompt_tokens` for the history truncation for the LLM, if history is used. Requires `DEPLOYMENT_NAME` model to support he history truncation and `max_prompt_tokens` parameter. Could be set to `0` to disable the history truncation for models which does not support it, but will cause error it if max model context window will be reached.

##### `DIAL_RAG__REQUEST__QA_CHAIN__CHAT_CHAIN__LLM__MAX_RETRIES`

*Optional*, default value: `2`

Sets the number of retries to send the request to the LLM.

##### `DIAL_RAG__REQUEST__QA_CHAIN__CHAT_CHAIN__SYSTEM_PROMPT_TEMPLATE_OVERRIDE`

*Optional*, default value: `None`

Allow to override the system prompt template.

##### `DIAL_RAG__REQUEST__QA_CHAIN__CHAT_CHAIN__USE_HISTORY`

*Optional*, default value: `True`

Used to set whether to use the history for the answer generation. If true, the previous messages from the chat history would be passes to the model. If false, only the query (last user message or standalone question, depending on the `query_chain` settings) will be passed to the model for the answer generation.

##### `DIAL_RAG__REQUEST__QA_CHAIN__CHAT_CHAIN__NUM_PAGE_IMAGES_TO_USE`

*Optional*, default value: `4`

Sets number of page images to pass to the model for the answer generation. If is greater that 0, the model in `llm.deployment_name` should accept images in the user messages. Could be set to 0 (together with USE_MULTIMODAL_INDEX=False and USE_DESCRIPTION_INDEX=False) for text-only RAG.

##### `DIAL_RAG__REQUEST__QA_CHAIN__CHAT_CHAIN__PAGE_IMAGE_SIZE`

*Optional*, default value: `1536`

Sets the size of the page images to pass to the model for the answer generation.

##### `DIAL_RAG__REQUEST__QA_CHAIN__QUERY_CHAIN__LLM__DEPLOYMENT_NAME`

*Optional*, default value: `gpt-4o-2024-05-13`

Used to set the deployment name of the LLM used in the chain. Could be useful if the model deployments have non-standard names in the Dial Core configuration.

##### `DIAL_RAG__REQUEST__QA_CHAIN__QUERY_CHAIN__LLM__MAX_PROMPT_TOKENS`

*Optional*, default value: `0`

Sets `max_prompt_tokens` for the history truncation for the LLM, if history is used. Requires `DEPLOYMENT_NAME` model to support he history truncation and `max_prompt_tokens` parameter. Could be set to `0` to disable the history truncation for models which does not support it, but will cause error it if max model context window will be reached.

##### `DIAL_RAG__REQUEST__QA_CHAIN__QUERY_CHAIN__LLM__MAX_RETRIES`

*Optional*, default value: `2`

Sets the number of retries to send the request to the LLM.

##### `DIAL_RAG__REQUEST__QA_CHAIN__QUERY_CHAIN__USE_HISTORY`

*Optional*, default value: `True`

Used to set whether to use the history for the chat history summarization to the standalone question for retrieval. If true, the previous messages from the chat history would be passes to the model to make a standalone question. If false, the last user message was assumed to be a standalone question and be used for retrieval as is.
<!-- generated-app-config-env-end -->

## Commands

Dial RAG supports following commands in messages:

### Attach

`/attach <url>` - allows to provide an url to the attached document in the message body. Is equivalent to the setting `messages[i].custom_content.attachments[j].url` in the [Dial API](https://epam-rail.com/dial_api#/paths/~1openai~1deployments~1%7BDeployment%20Name%7D~1chat~1completions/post).

The `/attach` command is useful to attach the document which is available in the Internet and is not uploaded to the Dial File Storage.


### Debug commands

Dial RAG supports following debug commands if the option `ENABLE_DEBUG_COMMANDS` is set to `true`.

 * `/model <model>` - allows to override the chat model used for the answer generation. Should be a deployment name of a chat model in available the Dial.
 * `/query_model <model>` - allows to override the model used to summarize the chat history to the standalone question. Should be a deployment name of a chat model in available the Dial. The model should support `tool calls`.
 * `/profile` - generates CPU profile report for the request. The report will be available as an attachment in the `Profiler` stage.


## Developer environment

This project uses [Python==3.11](https://www.python.org/downloads/) and [Poetry>=1.8.5](https://python-poetry.org/) as a dependency manager.

Check out Poetry's [documentation on how to install it](https://python-poetry.org/docs/#installation) on your system before proceeding.

If you have [Poetry>=1.8.5](https://python-poetry.org/) and python 3.11 installed in the system, to install requirements you can run:

```sh
poetry install
```

This will install all requirements for running the package, linting, formatting and tests.


Alternatively, if you have [uv](https://docs.astral.sh/uv/) installed, you can use it to create the environment with required version of Python and poetry:

```sh
uv venv "$VIRTUAL_ENV" --python 3.11
uvx poetry@1.8.5 install
```

This will install all requirements for running the package, linting, formatting and tests, the same as `poetry install` command above.

If you want to use poetry from the uv with make commands, you can set the `POETRY=uvx poetry@1.8.5` environment variable:

```sh
POETRY="uvx poetry@1.8.5" make install
```


### IDE configuration

The recommended IDE is [VSCode](https://code.visualstudio.com/).
Open the project in VSCode and install the recommended extensions.

This project uses [Ruff](https://docs.astral.sh/ruff/) as a linter and formatter. To configure it for your IDE follow the instructions in https://docs.astral.sh/ruff/editors/setup/.


### Make on Windows

As of now, Windows distributions do not include the make tool. To run make commands, the tool can be installed using
the following command (since [Windows 10](https://learn.microsoft.com/en-us/windows/package-manager/winget/)):

```sh
winget install GnuWin32.Make
```

For convenience, the tool folder can be added to the PATH environment variable as `C:\Program Files (x86)\GnuWin32\bin`.
The command definitions inside Makefile should be cross-platform to keep the development environment setup simple.

### Environment Variables

Copy `.env.example` to `.env` and customize it for your environment for the development process. See the [Configuration](#Configuration) section for the list of environment variables.

## Run

Run the development server locally:

```sh
make serve
```

Run the development server in Docker:

```sh
make docker_serve
```

Open `localhost:5000/docs` to make sure the server is up and running.

## Run with Dial in Docker Compose

The `docker_compose_local` folder contains the Docker Compose file and auxiliary scripts to run Dial RAG with [Dial Core](https://github.com/epam/ai-dial-core) in Docker Compose. The `docker-compose.yml` file is configured to run Dial RAG alongside [Dial Core](https://github.com/epam/ai-dial-core), [Dial Chat UI](https://github.com/epam/ai-dial-chat), and the [DIAL Adapter for DIAL](https://github.com/epam/ai-dial-adapter-dial) to provide access to LLMs.

### Steps to Configure and Run

1. In the `docker_compose_local` folder, create a file named `.env` and define the following variables:
    - `DIAL_RAG_URL` - Provide the URL for the local Dial RAG instance (including the IP address and port) if you are running it in your IDE. The default value is `http://host.docker.internal:5000`.
    - `REMOTE_DIAL_URL` - Provide the URL for the remote Dial Core to access the LLMs.
    - `REMOTE_DIAL_API_KEY` - Provide the API key for the remote Dial Core.
    - `DEPLOY_DIAL_RAG=<0|1>` - Set to `0` to skip deploying the Dial RAG container in Docker Compose (useful for debugging the application locally). Set to `1` to deploy Dial RAG as a Docker Compose container.

    These variables will be passed to `dial_conf/core/config.json` and used for communication between the Dial and Dial RAG applications.

2. Navigate to the `docker_compose_local` folder and run the following command in the terminal:

    ```sh
    docker-compose up
    ```

    This will bring up the entire Dial application, ready to use.

3. If you need to rebuild the Dial RAG image, use the following command:

    ```sh
    docker-compose up --build dial-rag
    ```


## Building docker file with predownloaded ColPali model
Due to large weight of each model, a separate docker image was created to avoid making the base image hold those weights when they are not needed.

`Dockerfile.colpali` - additional docker file that saves into the image one of the ColPali models.

There are a few arguments for building the image:

- `BASE_IMAGE_NAME` - argument that allows you to set the base image name for ai-dial-rag, default is `epam/ai-dial-rag:latest`
- `COLPALI_MODEL_NAME` - name of the ColPali model to download, default is `vidore/colSmol-256M`

And environment variable:
- `COLPALI_MODELS_BASE_PATH` - path where to store models inside the image, default is `/colpali_models`


## Lint

Run the linting before committing:

```sh
make lint
```

To auto-fix formatting issues run:

```sh
make format
```

## Test

Run unit tests locally:

```sh
make test
```

Run unit tests in Docker:

```sh
make docker_test
```

### Tests with cached LLM responses

Some of the tests marked with the `@e2e_test` decorator utilize cached results located in the `./tests/cache` directory. By default, these tests will use cached values. During test execution, you may encounter warning or failure messages such as `Failed: There is no response found in cache, use environment variable REFRESH=True to update` This indicates that some logic has changed and that the cached responses are out of date.

These tests can be executed using environment variables, or nox sessions:
- `make test` (or `nox -s test`) - usual test run, executed on CI. The test uses *ONLY* the cached responses from LLM. If cache missing, test throws an exception.
- `REFRESH=True make test` (or `nox -s test -- --refresh`) - This flag will delete all unused cache files, and stores new ones required by the executed tests.

To use the `REFRESH` flag, you need to have running dial-core on `DIAL_CORE_HOST` (default "localhost:8080") with `DIAL_CORE_API_KEY` (default "dial_api_key").


## Clean

To remove the virtual environment and build artifacts:

```sh
make clean
```

## Update docs

This project uses [settings-doc](https://github.com/radeklat/settings-doc) to generate the [Configuration](#Configuration) section of this documentation from the Pydantic settings.
To update the documentation run:

```sh
make docs
```

