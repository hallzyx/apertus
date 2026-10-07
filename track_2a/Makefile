.PHONY: run build test smoke
IMAGE ?= apertus-ost:local
PORT ?= 8000
PYTHON ?= python
DOCKER_BUILD_ARGS ?=
DOCKER_RUN_ARGS ?=
export DOCKER_CONFIG ?= /tmp/apertus-ost-docker
build:
	docker build $(DOCKER_BUILD_ARGS) -t $(IMAGE) .
run: build
	docker run --rm $(DOCKER_RUN_ARGS) -p $(PORT):8000 -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY -e OST_LABEL_MAP $(IMAGE) serve --host 0.0.0.0 --port 8000
test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
smoke: build
	docker run --rm $(IMAGE) self-test
