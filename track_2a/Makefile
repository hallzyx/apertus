.PHONY: run start build test smoke
RUNTIME ?= hybrid
IMAGE ?= apertus-ost:$(RUNTIME)
DOCKERFILE ?= $(if $(filter hybrid,$(RUNTIME)),Dockerfile.hybrid,$(if $(filter frozen,$(RUNTIME)),Dockerfile.frozen,Dockerfile))
MODEL_CACHE ?= $(if $(filter hybrid,$(RUNTIME)),apertus-retrieval-models,apertus-models)
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
	@if [ -n "$$HTTPS_PROXY" ] && command -v python >/dev/null && [ -z "$(DOCKER_RUN_ARGS)" ]; then bash scripts/cloud-docker.sh start; elif [ "$(RUNTIME)" = hybrid ]; then docker run --rm $(DOCKER_RUN_ARGS) -p $(PORT):8000 -v $(MODEL_CACHE):/models --add-host host.docker.internal:host-gateway -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY -e LLM_TIMEOUT_SECONDS -e RETRIEVAL_MODE $(IMAGE); elif [ "$(RUNTIME)" = frozen ]; then docker run --rm $(DOCKER_RUN_ARGS) -p $(PORT):8000 -v $(MODEL_CACHE):/models -e OST_LABEL_MAP $(IMAGE) $(FROZEN_ARGS); else docker run --rm $(DOCKER_RUN_ARGS) -p $(PORT):8000 --add-host host.docker.internal:host-gateway -e FROZEN_BASE_URL -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY -e LLM_TIMEOUT_SECONDS -e OST_LABEL_MAP $(IMAGE) serve --host 0.0.0.0 --port 8000; fi
test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
smoke: build
	docker run --rm $(if $(filter hybrid,$(RUNTIME)),--entrypoint python,) $(IMAGE) $(if $(filter hybrid,$(RUNTIME)),-m ost_nli self-test,$(if $(filter frozen,$(RUNTIME)),--self-test,self-test))
