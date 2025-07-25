PORT ?= 5001
IMAGE_NAME ?= ai-dial-rag
PLATFORM ?= linux/amd64
POETRY ?= poetry
DOCKER ?= docker
LIBREOFFICE_UBUNTU_VERSION ?= 4:24.2.7-0ubuntu0.24.04.4
ARGS ?=

# Check for CI environment
# Empty string means false in Makefile
CI ?=

.PHONY: all install build serve clean install_nox docs lint format install_libreoffice test docker_build docker_serve docker_test help

all: build

install:
	$(POETRY) install

build: install
	$(POETRY) build

serve: install
	$(POETRY) run uvicorn "aidial_rag.main:app" --reload --host "0.0.0.0" --port $(PORT) --env-file ./.env

clean: install_nox
	$(POETRY) run nox -s clean
	$(POETRY) env remove --all

install_nox:
	$(POETRY) install --only nox

docs: install_nox
	$(POETRY) run nox -s update_docs

lint: install_nox
	$(POETRY) run nox -s lint


format: install_nox
	$(POETRY) run nox -s format

install_libreoffice:
	@echo "Installing LibreOffice..."
	sudo apt-get update
	sudo apt-get install --no-install-recommends -y libreoffice=$(LIBREOFFICE_UBUNTU_VERSION)

test: install_nox $(if $(CI), install_libreoffice)
	$(POETRY) run nox -s test $(ARGS)

docker_build:
	$(DOCKER) build --platform $(PLATFORM) -t $(IMAGE_NAME):dev .

docker_serve: docker_build
	$(DOCKER) run --platform $(PLATFORM) --env-file ./.env --rm -p $(PORT):5000 $(IMAGE_NAME):dev

docker_test:
	$(DOCKER) build --platform $(PLATFORM) --target test .

help:
	@echo '===================='
	@echo "  make              - Default target, runs build"
	@echo "  make install      - Install project dependencies using Poetry"
	@echo "  make build        - Build the project package"
	@echo "  make serve        - Run the development server"
	@echo "  make clean        - Clean build artifacts and environments"
	@echo "  make install_nox  - Install nox dependencies"
	@echo "  make install_libreoffice - Install LibreOffice for CI"
	@echo "  make docs         - Update documentation"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make test         - Run tests"
	@echo "  make docker_build - Build Docker image"
	@echo "  make docker_serve - Run service in Docker"
	@echo "  make docker_test  - Run tests in Docker"
	@echo "  make help         - Display this help message"
