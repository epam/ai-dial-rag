PORT ?= 5001
IMAGE_NAME ?= ai-dial-rag
PLATFORM ?= linux/amd64
POETRY ?= poetry
DOCKER ?= docker
ARGS ?=

.PHONY: all install build serve clean install_nox docs lint format test docker_build docker_run

all: build


install:
	$(POETRY) install


build: install
	$(POETRY) build


serve: install
	$(POETRY) run uvicorn "aidial_rag.main:app" --reload --host "0.0.0.0" --port $(PORT) --env-file ./.env


clean:
	nox -s clean
	$(POETRY) env remove --all


install_nox:
	$(POETRY) install --only nox


docs: install_nox
	$(POETRY) run nox -s update_docs


lint: install_nox
	$(POETRY) run nox -s lint


format: install_nox
	$(POETRY) run nox -s format


test: install_nox
	$(POETRY) run nox -s test $(ARGS)


docker_build:
	$(DOCKER) build --platform $(PLATFORM) -t $(IMAGE_NAME):dev .


docker_serve: docker_build
	$(DOCKER) run --platform $(PLATFORM) --env-file ./.env --rm -p $(PORT):5000 $(IMAGE_NAME):dev


docker_test:
	$(DOCKER) build --platform $(PLATFORM) --target test .
