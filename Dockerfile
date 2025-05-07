FROM ubuntu:24.04 AS base

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV DEBIAN_FRONTEND=noninteractive

ENV BGE_EMBEDDINGS_MODEL_PATH=/embeddings_model/bge-small-en

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        ca-certificates \
        # Libreoffice is required for MS office documents
        libreoffice=4:24.2.7-0ubuntu0.24.04.3 \
        libmagic1 \
        # Dependency for opencv library
        libgl1 \
        # Install security fixes
        libc-bin=2.39-0ubuntu8.4 \
        libc6=2.39-0ubuntu8.4 \
        libgnutls30t64=3.8.3-1.1ubuntu3.3 \
        libtasn1-6=4.19.0-3ubuntu0.24.04.1 \
        libraptor2-0=2.0.16-3ubuntu0.1 \
        libgssapi-krb5-2=1.20.1-6ubuntu2.5 \
        libk5crypto3=1.20.1-6ubuntu2.5 \
        libkrb5-3=1.20.1-6ubuntu2.5 \
        libkrb5support0=1.20.1-6ubuntu2.5 \
        && \
    # Cleanup apt cache in the same command to reduce size
    apt-get clean && rm -rf /var/lib/apt/lists/*


FROM base AS builder

# Getting uv from distroless docker
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV VIRTUAL_ENV=/opt/venv

# Ubuntu 24.04 has python 3.12 by default
# We do not want to upgrade unstructured library for now,
# so we use uv to get python 3.11 while creating venv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv "$VIRTUAL_ENV" --python 3.11

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install pip requirements
COPY pyproject.toml poetry.lock ./

ENV POETRY=poetry@1.8.5
# uvx installs poetry in separate venv, not spoiling the app venv
RUN uvx "$POETRY" install --no-interaction --no-ansi --no-cache --only main --no-root --no-directory


FROM builder AS builder_download_nltk

# nltk 3.9 actually uses punkt_tab and averaged_perceptron_tagger_eng
# but we have to download punkt and averaged_perceptron_tagger as well, because unstructured will try to download it if missing
RUN python -m nltk.downloader -d /usr/share/nltk_data stopwords punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng


FROM builder AS builder_download_model

COPY download_model.py .

# Model: https://huggingface.co/epam/bge-small-en
RUN python download_model.py "epam/bge-small-en" "$BGE_EMBEDDINGS_MODEL_PATH" "openvino" "torch"


FROM builder AS builder_repo_digest

# Install git in builder to collect repository digest
RUN apt-get update && \
    apt-get install --no-install-recommends -y git

# Copy the whole repository
COPY . /opt/aidial_rag_repo
WORKDIR /opt/aidial_rag_repo

RUN python collect_repository_digest.py /opt/repository-digest.json


FROM builder_repo_digest as test

COPY --from=builder_download_nltk /usr/share/nltk_data /usr/share/nltk_data
COPY --from=builder_download_model "$BGE_EMBEDDINGS_MODEL_PATH" "$BGE_EMBEDDINGS_MODEL_PATH"

RUN uvx "$POETRY" install --no-interaction --no-ansi --no-cache --with test --no-directory
RUN uvx "$POETRY" run pytest tests


FROM base

WORKDIR /

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 1666 --disabled-password --gecos "" appuser
USER appuser

COPY --from=builder --chown=appuser /opt/uv/python /opt/uv/python
COPY --from=builder --chown=appuser /opt/venv /opt/venv
COPY --from=builder_download_nltk --chown=appuser /usr/share/nltk_data /usr/share/nltk_data
COPY --from=builder_download_model --chown=appuser "$BGE_EMBEDDINGS_MODEL_PATH" "$BGE_EMBEDDINGS_MODEL_PATH"
COPY --chown=appuser ./config /config
COPY --chown=appuser ./aidial_rag /aidial_rag
COPY --from=builder_repo_digest --chown=appuser /opt/repository-digest.json /opt/repository-digest.json

ENV PATH="/opt/venv/bin:$PATH"

# Disable usage tracking for unstructured
ENV DO_NOT_TRACK=true

# Currently you cannot pass shrink_factor from unstructured.partition to sort_page_elements
# default value 0.9 cuts parts of the tables in 10k pdf document
ENV UNSTRUCTURED_XY_CUT_BBOX_SHRINK_FACTOR=1.0

ENV DIAL_RAG__CONFIG_PATH=/config/azure_description.yaml
ENV DIAL_RAG__INDEX_STORAGE__USE_DIAL_FILE_STORAGE=False
ENV ENABLE_DEBUG_COMMANDS=False

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
EXPOSE 5000
CMD ["uvicorn", "aidial_rag.main:app", "--host", "0.0.0.0", "--port", "5000"]
