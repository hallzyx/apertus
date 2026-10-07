# Track 2A · OST booklet-grounded multilingual NLI

## Runtime

From the repository root, `make run` builds the pinned CPU Docker image and starts
port 8000. It downloads the public supporting retrieval model
`intfloat/multilingual-e5-small` at immutable revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`, verifying original LFS hashes.
The retrieval cache persists in Docker volume `apertus-retrieval-models`.
A practical retrieval runtime is four CPU threads, roughly 2 GB RAM and at least
3 GB free disk for Docker/retrieval weights. Apertus endpoint hardware is separate.
Internet is required for first asset download and remote model inference.

Set `LLM_NAME=swiss-ai/Apertus-v1.5-8B`, `LLM_BASE_URL` to an authorized
OpenAI-compatible endpoint ending `/v1`, and `LLM_API_KEY` securely if required.
The default runtime rejects model names that do not identify Apertus v1.5.
It does not silently substitute legacy Apertus 2509. Temperature is zero and output
is strictly parsed as a JSON integer class. Model token usage must come from the
server, not an estimate. Classification remains unavailable without a real endpoint.
`LLM_TIMEOUT_SECONDS` defaults to 120; increase it for slow CPU endpoints (for
example, 900). This is a client setting, not an organizer runtime limit.

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
docker run --rm -v apertus-retrieval-models:/models -v "$PWD/inputs:/inputs:ro" -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY apertus-ost:hybrid predict /inputs/booklet.pdf 'La proposta prevede un contributo annuo di 100 franchi.' --context hybrid --k 5
```

Host equivalent, after hash-locked CPU installation:

```bash
PYTHONPATH=track_2a/src EMBEDDING_MODEL_DIR=/workspace/.cache/multilingual-e5-small .venv/bin/python -m ost_nli predict /path/booklet.pdf 'The claim' --context hybrid --k 5
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
PYTHONPATH=track_2a/src EMBEDDING_MODEL_DIR=/workspace/.cache/multilingual-e5-small .venv/bin/python -m ost_nli predict-batch /inputs/claims.jsonl --output /outputs/predictions.json --context hybrid
```

Existing batch outputs are rejected. This same internal prediction function is
separable from the CLI and can be adapted to any later evaluator interface.

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

Once real v1.5 inference access exists, run matched full/reference/BM25/hybrid
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

All task-created Vast instances were destroyed and API absence verified. The
USD 10 hard budget and USD 2 reserve remain in `experiments/budget.json`.
Never rent before checking model access, stack support, current offer/cost and
bounded cleanup; persist verified results before destruction. No new GPU lease
was created for the current PDF/retrieval work.

Code/model licensing and additional results are in `technical_report.md`.
