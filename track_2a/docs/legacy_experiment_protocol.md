# Archived Apertus 2509 protocol

Historical snapshot only. Its model, evidence assumptions, pending questions and
lease state are superseded by challenge_contract.md and experiment_protocol.md.
Do not treat its scores as Apertus v1.5 or booklet-grounded performance.

# Updated contract and next real runs

Follow docs/challenge_contract.md. Production requires Apertus v1.5 and full
booklet + claim; old 2509 reference-only scores are historical diagnostics.
CLI/evidence schema are project choices, not blockers. Devpost is unnecessary.
Authenticate v1.5 weights or verify a v1.5 endpoint before any rental.
Use train-only fitting; choose architecture on validation, freeze it, and run
internal holdout NLI once. Real PDF retrieval diagnostics are committed separately.

The historical protocol below records earlier research; its scope ambiguity
was resolved by the supplied contract.

# Research protocol

## Status

Real original Apertus BF16 RTX 4090 inference has completed on all 276 strict
validation examples at 1,024/4,096-token provided-reference caps. Macro-F1 is
0.451653/0.453510. A train-only 30-row standardized option-logit classifier raises
these to 0.689366/0.743242; this passes the cheap frozen-decision GO gate.
The complete 902-training-row option/hidden cache and four grouped CPU heads
are verified. Selected hidden4096 scores 0.970353 (deduplicated 0.965299),
with training-only OOF temperature. Hidden1024 scores 0.958790 at about half
the encoder latency. All three task GPU leases are destroyed, API verified;
total observed credit reduction is USD 0.6372 (billing can settle asynchronously). Exact artifacts, failures and cleanup are in `experiments/`.
The final 310-row test is untouched. Official full/gold scope is still unresolved.

## First required runs

1. Field audit and provided-reference adapter completed. Confirm official class
   semantics, evidence interpretation and any required official evaluation protocol.
2. Strict frozen manifest completed: 902/276/310 examples. Review translated
   paraphrases and neutral claims' source booklet identities before training.
3. Obtain verified Apertus model ID/revision and inference stack. Inspect context
   window, quantization support, tokenizer, weights license and memory requirements.
4. Rent only after a current offer has been costed against the budget. Include
   downloads, warm-up, GPU rental, storage and transfer costs. Maintain a reserve.
5. Obtain full PDFs and gold annotations first. reference_string is a provided
   premise, not annotated minimal evidence. Run full context (A) and gold (B) on identical
   validation example IDs, prompt/class mapping, model revision and precision.
   Report document truncation and gold-empty counts. Use paired per-booklet bootstrap
   intervals before claiming a meaningful difference.
6. Record A vs B diagnostic: retrieval bottleneck only if evidence supports it;
   gold-context failure may instead imply NLI/prompt/data-label issues.
7. Measure BM25 and diversification next. Add multilingual dense/hybrid retrieval
   only when errors justify it; compare recall, F1, actual tokens and latency.
8. Compare prompted and constrained output; structured-output support must be
   verified, not silently replaced. Direct class-logit scoring requires a verified
   scoring path and tokenization of the full label alternatives.
9. Frozen option heads have measured improvements; hidden representations improve Macro-F1 to 0.958790/0.970353.
   Cache representations with dataset/model/prompt fingerprints before renting for
   another run. Train heads on CPU; calibrate using held-out training groups.
10. QLoRA and adaptive retrieval require a quantitative GO decision. The cheap head GO result does not require fine-tuning; the full frozen head is selected. No further QLoRA/70B/adaptive-routing spend is justified before official scope is resolved. At most two initial tuning runs if future evidence warrants them.

## Existing baseline runner

Run in `track_2a/`, with endpoint variables and verified mapping. The following
full/gold commands require actual full-booklet/gold-annotated inputs. The provided-
reference baseline currently available is documented in the README:

```bash
PYTHONPATH=src python -m ost_nli experiment FULL_BOOKLET_VALIDATION.jsonl --id A-full --split-name validation --context full --output-dir experiments/artifacts/A-full --gpu ACTUAL_GPU --estimated-cost ACTUAL_ESTIMATE
PYTHONPATH=src python -m ost_nli experiment GOLD_ANNOTATED_VALIDATION.jsonl --id B-gold --split-name validation --context gold --output-dir experiments/artifacts/B-gold --gpu ACTUAL_GPU --estimated-cost ACTUAL_ESTIMATE
PYTHONPATH=src python -m ost_nli experiment data/private/splits-strict/validation.jsonl --id C-bm25 --split-name validation --context bm25 --k 5 --output-dir experiments/artifacts/C-bm25 --gpu ACTUAL_GPU --estimated-cost ACTUAL_ESTIMATE
```

These commands are instructions for future real runs, **not executed experiments**.
Started/completed/failed events are append-only. Partial predictions survive a failed
run but are not scored as a completed run. Never overwrite an existing run directory.
Commit code before scientific runs; a dirty working tree is flagged in the registry.
Store actual model revision, GPU/stack and conclusion in experiment notes.

## Budget and lifecycle

`experiments/budget.json` starts at USD 10 and records current reservations and observed credit reduction.
Read timestamped active lease/cleanup observations; billing may settle asynchronously.

Use the official Vast API `https://console.vast.ai/api/v0/` (paginated instance lists use `/api/v1/instances/`), with the secret injected
as `VAST_API_KEY`. Keys must never be printed or stored. Proxy-bound credentials
must stay in Authorization headers to their declared destination, not SSH commands
or request query strings. Before creation inspect account balance, existing instances,
offer hourly price, VRAM, reliability, bandwidth and storage rate. Record the
instance and spending reservation immediately, with a conservative runtime limit.

Stop rental work before the conservative cost bound crosses USD 10, including storage
and transfer. Persist results to the repository or verified durable storage **before**
destroying the instance. Destroy (not merely stop) task-created instances and query
the API until their absence is confirmed. Do not destroy pre-existing user instances.
Without an accessible account, no rental or account-wide cleanup claim is permitted.
