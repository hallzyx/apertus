# Track 2A · OST booklet-grounded multilingual NLI

## Runtime

From the repository root, `make run` builds the pinned PDF/HTTP Docker image and starts port 8000. `deployment/v15-selection.json` supplies the validation-selected context; currently this is full/capped booklet text, with a 48,000-byte document-context budget and explicit truncation reporting. No supporting retrieval model is downloaded by the default frontend. Apertus endpoint hardware and weights are separate.

Optional retrieval runs use `make run RUNTIME=hybrid RETRIEVAL_MODE=hybrid` (or `dense`). That image loads the public supporting `intfloat/multilingual-e5-small` model at revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, verifying original hashes and retaining assets in volume `apertus-retrieval-models`. Full/BM25 modes skip E5. Retrieval research uses four CPU threads; VFS cloud Docker can require around 8 GB free during a retrieval-image build. Classification requires the configured Apertus endpoint.

Set `LLM_NAME=swiss-ai/Apertus-v1.5-8B`, `LLM_BASE_URL` to an authorized
OpenAI-compatible endpoint ending `/v1`, and `LLM_API_KEY` securely if required.
The default runtime rejects model names that do not identify Apertus v1.5.
It does not silently substitute legacy Apertus 2509. Temperature is zero and output
is strictly parsed as a JSON integer class. Model token usage must come from the
server, not an estimate. Classification remains unavailable without a real endpoint.
`LLM_TIMEOUT_SECONDS` defaults to 120; increase it for slow CPU endpoints (for
example, 900). This is a client setting, not an organizer runtime limit.

For Blackwell GPUs, install `requirements-v15-cuda.lock` in the pinned Python 3.11
Linux runtime before `requirements-v15.lock`. Torch, TorchAudio and TorchVision
must all use CUDA 12.8; the base image
`pytorch/pytorch@sha256:dab81780fd94483b67b4b5679cc0024939b08e48540d39476d284cb29002ed69`
contains CUDA 12.6 extensions, so CUDA availability alone is insufficient. Run
`scripts/check_v15_cuda.py` and verify actual BF16/recurrent kernels, then native
model/embedding loading. CPU uses its separately pinned CPU dependencies.

The cloud proxy helper preserves TLS/CA verification. Do not expose keys in Git,
logs or commands. Optional direct local v1.5 serving requires its gated weights
and verified native architecture support; the old frozen backend cannot load its
`apertus1p5` architecture or reuse its decision head.

After securely configuring `HF_TOKEN` for `huggingface.co`, check access without
downloading weights or renting a GPU:
`../.venv/bin/python scripts/check_v15_access.py` from `track_2a/`.
This checks the pinned model configuration and reports installed architecture
support without printing the credential.

## Application and deterministic CLI

Upload a PDF in the Docker UI, or supply canonical document JSON. Production uses
only document text and claim. Returned labels are fixed:
0 entailment, 1 neutral, 2 contradiction. Source evidence contains one-based page,
exact extracted text, paragraph ID, source PDF SHA-256 and page-text character offsets.
Several retrieved passages are allowed; they are candidate supporting context,
not a claim of perfectly judged or uniquely correct evidence. PDFs with no selectable
text need OCR; no evidence is fabricated.

With Docker running and cached assets, execute:

```bash
# Mount your booklet and configure the same authorized endpoint.
docker run --rm -v "$PWD/inputs:/inputs:ro" -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY -e LLM_TIMEOUT_SECONDS apertus-ost:api predict /inputs/booklet.pdf 'La proposta prevede un contributo annuo di 100 franchi.' --k 5
```

Host equivalent, after hash-locked PDF installation (uses the selected full/capped context):

```bash
PYTHONPATH=track_2a/src .venv/bin/python -m ost_nli predict /path/booklet.pdf 'The claim'
```

Output includes `label`, `label_name`, `evidence`, `input_tokens`,
`inference_time_ms`, model name and truncation flags. `probabilities` is null when
the endpoint supplies only a class; no probabilities are synthesized.
`model_inference_time_ms` separates the model call from overall retrieval + NLI time.
PDF parsing/model initialization are separate startup/input-preparation costs.
Exit code 0 means success; invalid input/unavailable inference exits 2. This is
our documented interface, not an organizer-mandated JSON or CLI signature.

Batch input is JSONL with unique `id`, `document` path and `claim`, no labels or
reference strings. Relative document paths resolve against the input file directory:

```json
{"id":"example-1","document":"booklet.pdf","claim":"The claim"}
```

```bash
PYTHONPATH=track_2a/src .venv/bin/python -m ost_nli predict-batch /inputs/claims.jsonl --output /outputs/predictions.json
```

Existing batch outputs are rejected. This same internal prediction function is
separable from the CLI and can be adapted to any later evaluator interface.

## Reproduce the selected Apertus classifier

The selected system keeps all Apertus weights frozen and applies a small trained
classifier to the model's last-token hidden representation. It is not LoRA.
The head is `deployment/head-v15.json`; the selection and quantitative results
are recorded in `deployment/v15-selection.json` and `technical_report.md`.
A generic base-model chat endpoint does not reproduce this classifier.

Use Python 3.11/Linux in a separate native environment. Install
`requirements-v15-cuda.lock`, `requirements-v15.lock`, then
`requirements-head-v15.lock` with
`pip install --require-hashes`, and run `scripts/check_v15_cuda.py`.
The measured validation and holdout used A40 and RTX A6000 GPUs with 48 GB VRAM;
these are reported experiment hardware, not an organizer-mandated minimum.
The ordinary application `.venv` does not imply native v1.5 architecture support.

Download the pinned model once using authorized `HF_TOKEN`, then verify and serve
its local cache. From the repository root, with the native environment active:

```bash
export APERTUS_MODEL_DIR=/workspace/.cache/apertus-v15-model
PYTHONPATH=track_2a/src HF_HUB_DISABLE_XET=1 python - <<'PY'
import os
from ost_nli.v15 import download
download(os.environ['APERTUS_MODEL_DIR'], os.environ['HF_TOKEN'])
PY
PYTHONPATH=track_2a/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python track_2a/scripts/serve_v15.py \
  --model-dir "$APERTUS_MODEL_DIR" --device cuda \
  --method head --head track_2a/deployment/head-v15.json --port 8001
```

The download helper verifies the original pinned revision and all six weight
shard hashes; the server rechecks its manifest before loading. Weights total
about 18.4 GB and must remain outside Git. Cached inference does not need a token.
Wait for `V15_SERVING_READY` and verify the backend's `/health` reports
`method: head`. The service listens on loopback.

On Linux, connect the Docker frontend to that host backend in another terminal:

```bash
export LLM_NAME=swiss-ai/Apertus-v1.5-8B
export LLM_BASE_URL=http://127.0.0.1:8001/v1
export LLM_API_KEY=
unset FROZEN_BASE_URL
make run DOCKER_RUN_ARGS='--network host -e NO_PROXY=127.0.0.1,localhost'
```

For another hosting arrangement, set a reachable authorized `/v1` endpoint and
its required API credential securely. Earlier CPU smoke checks used `--device cpu`
with a separately pinned CPU Torch installation, but CPU inference is slow and
the reported benchmark latencies are GPU measurements. Increase client timeout
when appropriate. Starting the frontend without a backend still supports PDF
processing and source inspection; classification fails explicitly.

## Reproduce official booklet preparation and diagnostics

```bash
bash track_2a/scripts/setup-cloud.sh
PYTHONPATH=track_2a/src .venv/bin/python track_2a/scripts/prepare_booklets.py --source track_2a/data/private/official/v1.1.jsonl --splits track_2a/data/private/splits-strict --output track_2a/data/private/full-booklets-v2
PYTHONPATH=track_2a/src HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from ost_nli.dense import download; download('/workspace/.cache/multilingual-e5-small')"
PYTHONPATH=track_2a/src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python track_2a/scripts/evaluate_booklet_retrieval.py --full track_2a/data/private/full-booklets-v2/validation.jsonl --reference track_2a/data/private/splits-strict/validation.jsonl --embedding-dir /workspace/.cache/multilingual-e5-small --cache-dir /workspace/.cache/e5-document-vectors --output /tmp/new-retrieval-validation
```

60 official PDFs cover all 902/276/310 rows. Source URL/hash manifests and alignment
are in `experiments/booklet-source-v2/`. The 5-gram overlap diagnostic is not the
organizer's unspecified evidence score or exact gold Recall@k. Reference text is
used only by the evaluator after retrieval, never by the retriever or production.
The 310 split is our internal holdout, not the organizer's hidden benchmark.

With the verified v1.5 native stack or authorized endpoint, run matched full/reference/BM25/hybrid
experiments using the committed strict IDs. Reference inputs are diagnostics only:

```bash
cd track_2a
PYTHONPATH=src ../.venv/bin/python -m ost_nli experiment data/private/full-booklets-v2/validation.jsonl --id v15-hybrid-validation --split-name validation --context hybrid --output-dir experiments/artifacts/v15-hybrid-validation
```

Set `EMBEDDING_MODEL_DIR` to the verified local E5 directory for this host command.
The generic evaluator computes fixed three-class Macro-F1, class/language/pair/
cross-lingual slices, actual tokens, latency and calibration only when real
probabilities exist. Logs preserve started/completed/failed runs and partial results.

## Legacy research and budget

`make run RUNTIME=frozen` remains an optional original Apertus 2509 research
runtime, requiring 32 GB RAM and about 20 GB disk for 16.1 GB weights. It is not
compliant with the v1.5 generation requirement. Historical 0.970353 results are
reference-only; see `docs/legacy_apertus_2509_report.md`. Binary feature caches are
ignored in the current checkout and must be regenerated for those old experiments.

Task-created Vast research leases and verified destruction are recorded in
`experiments/budget.json`; inspect `active_instance_ids` for current status. The
USD 10 hard budget and USD 2 reserve apply to every retry.
Never rent before checking model access, stack support, current offer/cost and
bounded cleanup; persist verified results before destruction. Real v1.5 GPU research is separately budgeted and guarded; CPU PDF/retrieval
checks are not GPU benchmark results.

Code/model licensing and additional results are in `technical_report.md`.
