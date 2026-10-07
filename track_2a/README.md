# Apertus Evidence Lab

Track 2A · OST · Multilingual Natural Language Inference over Swiss Official Voting Booklets.
The official template's track layout is retained. Original track instructions are
in `docs/official_track_readme.md`.

## Run through Docker

Requires Docker with BuildKit and Make. No host Python dependencies are needed for
ordinary `make run`. The image includes Python 3.11 and hash-verified pypdf for
text-based PDF input. The base image is pinned by digest.

```bash
make run
```

This starts the workbench on port 8000. Without a model it supports real lexical
retrieval and clearly reports classification as unavailable. Configure an existing
Apertus OpenAI-compatible endpoint for inference:

- `LLM_NAME`: exact Apertus model ID/version served by the endpoint.
- `LLM_BASE_URL`: HTTPS endpoint base, conventionally ending `/v1`.
- `LLM_API_KEY`: secure endpoint authentication, if required; never commit it.
- `OST_LABEL_MAP`: JSON mapping official keys `"0"`, `"1"`, `"2"` to the verified
  class names `entailment`, `contradiction`, `neutral`. The source dataset card does
  not document these semantics; there is deliberately no guessed default.

An endpoint name being configured does not prove connectivity. The workbench
never fabricates labels, probabilities, usage or accuracy. Probabilities are null
for generation methods; probability calibration is available in the evaluator
when real class probabilities exist.

In Codex Cloud's proxied Docker environment use:

```bash
bash scripts/cloud-docker.sh run
```

This resolves the current proxy hostname and supplies the environment's trusted
CA bundle to Docker via supported build secrets/runtime mounts. No fixed IP,
proxy credential or verification bypass is saved. Outside this environment,
ordinary Docker networking and public CAs work with `make run`.

## CLI

### Optional real Apertus inference on CPU

The original public Apertus 8B model has executed on CPU. This optional workflow
requires about 16.1 GB of weights, additional dependency/cache space, and at least
32 GB RAM. It is separate from the lightweight Docker workbench. From the
repository root:

```bash
UV_CACHE_DIR=/workspace/.cache/uv uv pip install --python .venv/bin/python --torch-backend cpu --require-hashes -r track_2a/requirements-cpu.lock
.venv/bin/python track_2a/scripts/download_cpu_model.py --model-dir /workspace/.cache/apertus-8b
.venv/bin/python track_2a/scripts/cpu_apertus_smoke.py --model-dir /workspace/.cache/apertus-8b --output-dir /tmp/apertus-new-smoke
PYTHONPATH=track_2a/src .venv/bin/python track_2a/scripts/cpu_ost_experiment.py --model-dir /workspace/.cache/apertus-8b --train track_2a/data/private/splits-strict/train.jsonl --validation track_2a/data/private/splits-strict/validation.jsonl --output-dir track_2a/experiments/new-cpu-reference-run --train-per-class 10 --reference-token-cap 1024
```

Existing nonempty experiment directories are rejected. The completed short
synthetic diagnostic got 2/3 correct, including a contradiction error; it is not
an OST benchmark. The long-reference OST run is recorded separately and its
`experiment.json` status must be `completed` before treating its metrics as a
full validation result. The numeric option mapping uses training labels only;
it does not establish the official class-name convention. CPU prefill on this
machine can take 15–40 seconds per long example.

```bash
PYTHONPATH=src python -m ost_nli retrieve data/example-booklet.json 'jährlicher Beitrag 100 Franken'
PYTHONPATH=src python -m ost_nli predict /path/to/booklet.pdf 'natural-language claim'
```

JSON booklets contain `document_id` and `passages` with unique `id`, `text`, and
optional one-based `page` and `language`. Text files retain paragraph provenance
with unknown pages marked null. PDFs with no selectable text fail clearly; OCR
is not implemented. The Docker CLI can process host files mounted read-only:

```bash
docker run --rm -v "$PWD/data:/input:ro" -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY -e OST_LABEL_MAP apertus-ost:local predict /input/example-booklet.json 'claim'
```

Inference returns integer class, class name, available probabilities, selected
passages, page/source provenance, server-reported prompt tokens and measured
end-to-end latency. Classification is independent of explanations.

## Official data and evaluation

```bash
PYTHONPATH=src python -m ost_nli prepare-ost --output data/private/ost-reference.jsonl
PYTHONPATH=src python -m ost_nli split data/private/ost-reference.jsonl --output-dir data/private/splits-strict --seed 42
PYTHONPATH=src python -m ost_nli inspect data/private/ost-reference.jsonl
```

The downloader pins OST dataset revision
`9ff08597fb79dc68cbb3af9eb1388f34d21223e6` and verifies its published Git LFS SHA-256.
Existing changed source files and existing output/split directories are rejected,
not silently overwritten. The dataset provides references, not minimal gold
evidence or full PDF text. The adapter marks this distinction explicitly.

Default splits keep voting dates together and connect dates that share normalized
identical claims. Translations and near-duplicates still require a source audit.
The current frozen split sizes are 902 train / 276 validation / 310 test. No
official final test performance has been measured.

```bash
PYTHONPATH=src python -m ost_nli experiment data/private/splits-strict/validation.jsonl --id reference-prompt-v1 --split-name validation --context reference --output-dir experiments/artifacts/reference-prompt-v1 --notes 'record exact model revision, quantization and stack'
PYTHONPATH=src python -m ost_nli evaluate data/private/splits-strict/validation.jsonl experiments/artifacts/reference-prompt-v1/predictions.json
```

Real experiments require the endpoint and verified label mapping. Dataset hashes
and exact ID alignment prevent stale/mismatched predictions from being scored.
The append-only registry preserves failed runs. Per-language and cross-lingual
Macro-F1, token/latency coverage and calibration availability are reported.
`--context gold` requires actual gold annotations; `--context full` refuses the
provided-reference dataset rather than misreporting it as full-booklet context.

## Validation and budget

```bash
make test
make smoke                 # tests inside Docker
bash scripts/cloud-docker.sh smoke  # cloud proxy variant
```

The synthetic example is a UI/test fixture, not official benchmark data.
`experiments/budget.json` records the USD 10 ceiling, observed spend, active leases and verified cleanup.
Read its timestamped status before any new rental; billing can settle asynchronously.
Never rent before costing GPU, storage, transfer and download/warm-up time.

## Licensing and provenance

Application code: Apache-2.0, inherited official project template license.
Official template: `HackApertus/project-template`, revision
`7f2382275461baf3fa6c8855d157d86abffe9f0e`.
OST data: dataset metadata declares MIT; data downloaded separately, attributed
to `OSTswiss/MNLIoverSwissVotingBooklets`. Source cards and research limitations
are recorded in `docs/` and the technical report.
pypdf is BSD-3-Clause; its license is included in the installed distribution.
Final submission licensing remains subject to the official event terms.


## Real frozen Apertus decision heads

Completed real internal validation experiments and per-language/calibration metrics
are committed in `experiments/` and summarized in `technical_report.md`. The current
30-training-row option head reaches Macro-F1 0.743242 at a 4,096-token reference cap.
This is internal supplied-reference validation, not an official challenge score.

After the optional original-model CPU installation/download above, classify a new
JSON/text/PDF directly from the repository root:

```bash
PYTHONPATH=track_2a/src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python -m ost_nli predict-frozen track_2a/data/example-booklet.json 'La proposta prevede un contributo annuo di 100 franchi.' --model-dir /workspace/.cache/apertus-8b --head-dir track_2a/experiments/apertus-option-head-4096-v1
```

The head uses the exact official chat template and frozen BF16 model. It returns
class 0/1/2, genuine classifier probabilities, exact model-input premise, available
page provenance, tokenizer context tokens and measured latency. Official class names
remain null without verified `OST_LABEL_MAP`; an explicitly marked training-inferred
semantic name is separate. The 30-row head is uncalibrated. Full-booklet or BM25
inputs are accepted for exploration but their challenge accuracy has not been measured.

For a persistent local model serving the Docker workbench, start this process
before `make run` in a second terminal (32 GB RAM and original weights required):

```bash
PYTHONPATH=track_2a/src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python track_2a/scripts/serve_frozen_apertus.py --model-dir /workspace/.cache/apertus-8b --head-dir track_2a/experiments/apertus-option-head-4096-v1 --host 0.0.0.0 --port 8001
FROZEN_BASE_URL=http://host.docker.internal:8001 make run
```

The backend serializes inference to avoid concurrent memory pressure. Its `/health`
identifies the loaded model/head; only a successful `/api/predict` proves connectivity.
Keep this local service on a trusted development network. Default `make run` remains
a lightweight workbench; an authorized Apertus endpoint can also use the documented
OpenAI-compatible path. No inference credential is required for local frozen inference.

For CPU training on recovered frozen caches, install:

```bash
UV_CACHE_DIR=/workspace/.cache/uv uv pip install --python .venv/bin/python --require-hashes -r track_2a/requirements-head.lock
PYTHONPATH=track_2a/src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python track_2a/scripts/frozen_head_experiment.py --cache track_2a/experiments/apertus-frozen-cache-v1 --cap 4096 --features option_logits --train track_2a/data/private/splits-strict/train.jsonl --validation track_2a/data/private/splits-strict/validation.jsonl --output-dir /tmp/new-frozen-head
```

The training script checks cache/data hashes and exact row IDs; grouped OOF training
selects C and fits temperature without validation labels. Never overwrite completed
outputs. `vast_feature_cache.sh` extracts the two context caps on a properly costed
original-model CUDA lease; `recover_vast_cache.py` recovers checksummed UTF-8 parts
from that task's stopped instance. Persist recovered results and destroy/verify the
lease; stopping alone leaves billable storage.
