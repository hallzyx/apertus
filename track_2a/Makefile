.PHONY: run start build test smoke
RUNTIME ?= frozen
IMAGE ?= apertus-ost:$(if $(filter frozen,$(RUNTIME)),frozen,local)
DOCKERFILE ?= $(if $(filter frozen,$(RUNTIME)),Dockerfile.frozen,Dockerfile)
MODEL_CACHE ?= apertus-models
FROZEN_ARGS ?=
PORT ?= 8000
PYTHON ?= $(if $(wildcard ../.venv/bin/python),../.venv/bin/python,python)
DOCKER_BUILD_ARGS ?=
DOCKER_RUN_ARGS ?=
export DOCKER_CONFIG ?= /tmp/apertus-ost-docker
build:
	@if [ -n "$$HTTPS_PROXY" ] && command -v python >/dev/null && [ -z "$(DOCKER_BUILD_ARGS)" ]; then bash scripts/cloud-docker.sh build; else docker build $(DOCKER_BUILD_ARGS) -f $(DOCKERFILE) -t $(IMAGE) .; fi
run: build
	$(MAKE) start
start:
	@if [ -n "$$HTTPS_PROXY" ] && command -v python >/dev/null && [ -z "$(DOCKER_RUN_ARGS)" ]; then bash scripts/cloud-docker.sh start; elif [ "$(RUNTIME)" = frozen ]; then docker run --rm $(DOCKER_RUN_ARGS) -p $(PORT):8000 -v $(MODEL_CACHE):/models -e OST_LABEL_MAP $(IMAGE) $(FROZEN_ARGS); else docker run --rm $(DOCKER_RUN_ARGS) -p $(PORT):8000 --add-host host.docker.internal:host-gateway -e FROZEN_BASE_URL -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY -e OST_LABEL_MAP $(IMAGE) serve --host 0.0.0.0 --port 8000; fi
test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
smoke: build
	docker run --rm $(IMAGE) $(if $(filter frozen,$(RUNTIME)),--self-test,self-test)
