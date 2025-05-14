PORT ?= 5001
IMAGE_NAME ?= ai-dial-rag
PLATFORM ?= linux/amd64
POETRY ?= poetry
DOCKER ?= docker
LIBREOFFICE_UBUNTU_VERSION ?= 4:24.2.7-0ubuntu0.24.04.4
ARGS ?=
# Detect OS
OS := $(shell uname -s 2>/dev/null || echo "Windows")
# Check for CI environment
CI ?= false

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
ifeq ($(OS),Linux)
	sudo apt-get update
	sudo apt-get install --no-install-recommends -y libreoffice=$(LIBREOFFICE_UBUNTU_VERSION)
else ifeq ($(OS),Darwin)
	brew install --cask libreoffice
else ifeq ($(OS),Windows)
	@powershell.exe -Command "if (Get-Command winget -ErrorAction SilentlyContinue) { winget install -e --id TheDocumentFoundation.LibreOffice } else { Write-Host 'Error: winget not found. Please install LibreOffice manually from https://www.libreoffice.org/download/download/' }"
else
	@echo "Can't install LibreOffice automatically, please check https://www.libreoffice.org/download for manual installation."
endif

test: install_nox
ifeq ($(CI),true)
	@echo "CI environment detected"
	$(MAKE) install_libreoffice
else
	@if command -v libreoffice >/dev/null 2>&1 || [ "$(OS)" = "Windows" ] && (powershell.exe -Command "if (Get-Command soffice -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"); then \
		echo "LibreOffice found, proceeding with tests..."; \
	else \
		echo "ERROR: LibreOffice not found. Tests require LibreOffice for document processing."; \
		echo "Please run 'make install_libreoffice' or install manually."; \
		exit 1; \
	fi
endif
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
	@echo "  make install_libreoffice - Install LibreOffice for document processing"
	@echo "  make docs         - Update documentation"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make test         - Run tests"
	@echo "  make docker_build - Build Docker image"
	@echo "  make docker_serve - Run service in Docker"
	@echo "  make docker_test  - Run tests in Docker"
	@echo "  make help         - Display this help message"
