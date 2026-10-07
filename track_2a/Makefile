.PHONY: run start build test smoke
IMAGE ?= apertus-ost:local
PORT ?= 8000
PYTHON ?= $(if $(wildcard ../.venv/bin/python),../.venv/bin/python,python)
DOCKER_BUILD_ARGS ?=
DOCKER_RUN_ARGS ?=
export DOCKER_CONFIG ?= /tmp/apertus-ost-docker
build:
	@if [ -n "$$HTTPS_PROXY" ] && command -v python >/dev/null && [ -z "$(DOCKER_BUILD_ARGS)" ]; then bash scripts/cloud-docker.sh build; else docker build $(DOCKER_BUILD_ARGS) -t $(IMAGE) .; fi
run: build
	$(MAKE) start
start:
	@if [ -n "$$HTTPS_PROXY" ] && command -v python >/dev/null && [ -z "$(DOCKER_RUN_ARGS)" ]; then bash scripts/cloud-docker.sh start; else docker run --rm $(DOCKER_RUN_ARGS) -p $(PORT):8000 -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY -e OST_LABEL_MAP $(IMAGE) serve --host 0.0.0.0 --port 8000; fi
test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
smoke: build
	docker run --rm $(IMAGE) self-test
