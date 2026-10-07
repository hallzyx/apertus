# Historical research before the clarified v1.5/full-booklet contract

# Technical report — Apertus Evidence Lab

Track 2A · OST · Hack Apertus. Research results on a frozen internal validation split; official submission compliance is pending the challenge specification.

## 1. Result and architecture

The selected system scores **0.970353 Macro-F1** on the 276-row strict internal supplied-reference validation split, compared with **0.453510** for raw original Apertus option scoring at the same 4,096-token cap. Exact-deduplicated validation (207 rows) scores **0.965299**. The final 310-row test is untouched; these are not official challenge results.

Pipeline: supplied reference and multilingual claim → official Apertus chat template → original BF16 frozen inference → 4,096-dimensional final normalized hidden state at the last assistant-prefix token → training-only standardized multinomial L2 classifier → training-group OOF scalar temperature → numeric class. The selected head uses all 902 training rows, C=1 and T=1.415300. No replacement language model or language-model weight updates are used. Classification does not require explanation generation.

The complete CUDA cache contains all 902 training and 276 validation examples at both context caps. Four CPU experiments compare option logits with hidden states. Regularization is selected exclusively by connected-group training OOF Macro-F1; temperature minimizes training OOF NLL. Architecture selection uses internal validation Macro-F1. Only two independent training components and two validation components limit confidence.
The selected model makes eight validation errors (DE 4 / FR 1 / IT 3), retained with numeric targets, predictions and probabilities in `apertus-frozen-hidden-4096-v1/error_audit.json`. Portable deployment probabilities match all four recorded heads on all 276 rows within 1e-10; original archive, part and matrix source-side SHA-256 checks passed.

The public model is `swiss-ai/Apertus-8B-Instruct-2509`, immutable revision `b946d40447b2b597999b9c86d44bee0b452c919f`. All four original BF16 shards are verified against official Git LFS SHA-256 metadata. GPU stack: Torch 2.8.0+cu126, Transformers 4.56.2, SDPA, native PyTorch XIELU fallback, seed 42. Pinned image digest and exact source commits are in experiment records. No 70B, A100/H100 or QLoRA expenditure is justified yet.

## 3. Official dataset and methodology

Source: [OSTswiss/MNLIoverSwissVotingBooklets](https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets).
Pinned revision: `9ff08597fb79dc68cbb3af9eb1388f34d21223e6`.
JSONL SHA-256: `87ee4afd9c0ee8861106c50f05884481402ca668aecf927df68fc52d915b02fd`.
Source metadata declares MIT. The complete card currently contains only license
metadata; it does not define numeric class semantics or official scoring protocol.

| Property | Observed value |
|---|---:|
| Examples | 1,488 |
| Voting dates / source booklet groups | 20 |
| Language-specific PDF URLs | 60 |
| Distinct provided reference strings | 667 |
| Classes 0 / 1 / 2 | 495 / 498 / 495 |
| Claims DE / FR / IT | 495 / 497 / 496 |
| References DE / FR / IT | 455 / 569 / 464 |
| Exact duplicate claim/reference/language/label examples | 335 |
| Normalized claims repeated across dates | 12 |
| Claim word count: min / median / max | 6 / 21 / 63 |

Word counts use Unicode word segmentation, **not model tokens**. All nine language
pairs occur. Counts by pair are in `experiments/dataset_audit.json`.

Fields observed: claim, claim_language, reference_string, entailment_label,
baseline_score, reference_language, booklet_publish_date, booklet_download_date,
booklet_url, vote. `baseline_score` is excluded from inputs/features and evaluation;
its origin and meaning are undocumented, and using it risks label leakage.

The canonical adapter retains the provided premise, source URL and original row
index. Reference paragraphs have no fabricated page numbers. `evidence_ids=[]`
and `evidence_annotation_available=false` explicitly mark missing minimal-gold
annotations. Provided-reference paragraphs are not called gold evidence. Their
word length is not represented as an observed gold-evidence length.

The source's neutral examples may concern claims about other referendum proposals.
Expanding from the given reference to a full booklet could change the intended
relation. The official task must clarify the admissible premise scope before
interpreting full-context versus reference-context scores as retrieval improvement.

## 4. Leakage-aware frozen partitions

First group all language variants by booklet publication date. Connect date groups
sharing a normalized identical claim. This produces seven connected components.
Assign components by descending size to training/validation/test using target count
deficits, with seed-42 hash tie breaks. No label-based seed or partition search.

| Partition | Examples | Dates | Classes 0 / 1 / 2 |
|---|---:|---:|---|
| Train | 902 | 13 | 317 / 296 / 289 |
| Validation | 276 | 3 | 86 / 98 / 92 |
| Test | 310 | 4 | 92 / 104 / 114 |

The full manifest, row IDs, fingerprints and distributions are committed at
`experiments/split_manifest.json`. Dates and normalized identical claims do not
cross partitions. Translated/paraphrased source claims and document editions still
need review: these controls do not prove zero semantic leakage. Exact duplicates
are retained to preserve the source population; report deduplicated sensitivity
results before claiming robust model improvements. With only three validation
dates, uncertainty must be assessed at booklet level, not by treating rows as
independent observations.


## 5. Evaluation and comparisons

Macro-F1 averages fixed classes 0/1/2; undefined class F1 is zero. Exact prediction IDs and dataset fingerprints are required. The evaluator also reports per-class F1, confusion matrices, DE/FR/IT and language-pair slices, cross-lingual performance, context tokens, latency, ECE with ten equal-width bins, multiclass Brier and NLL when real probabilities exist. Missing gold evidence metrics remain null.

All complete real OST comparisons below use the same 276 validation rows. The final 310-row test partition remains untouched. An exact-duplicate sensitivity check retains 207 validation rows: raw 4,096-token Macro-F1 is 0.488089 and its 30-row head is 0.739191; see `experiments/exact_duplicate_sensitivity.json`. Context tokens include the system, claim and official chat template. GPU latency covers tokenization and inference; head latency adds a separately timed CPU head, and is not a measurement of an integrated live service.

| Experiment | Validation Macro-F1 | Mean context tokens | Mean inference seconds |
|---|---:|---:|---:|
| cpu-majority-v1 | 0.158379 | 0 | 3.1203985616335506e-07 |
| apertus-8b-gpu-reference-1024-v1 | 0.451653 | 983.6884057971015 | 0.15632816332230426 |
| apertus-8b-gpu-reference-4096-v1 | 0.453510 | 1996.1485507246377 | 0.32803798428115744 |
| apertus-option-head-1024-v1 | 0.689366 | 983.6884057971015 | 0.15646711742012973 |
| apertus-option-head-4096-v1 | 0.743242 | 1996.1485507246377 | 0.32822530838622854 |
| apertus-frozen-option-1024-v1 | 0.723639 | 983.688406 | 0.156522 |
| apertus-frozen-option-4096-v1 | 0.757375 | 1996.148551 | 0.323269 |
| apertus-frozen-hidden-1024-v1 | 0.958790 | 983.688406 | 0.156550 |
| apertus-frozen-hidden-4096-v1 | 0.970353 | 1996.148551 | 0.323438 |

The raw baselines fit the semantic A/B/C to numeric correspondence exclusively on 30 training rows: [0,2,1]. This is supervised correspondence, not independent verification of official class names. Raw option scoring retains restricted logits and full-vocabulary greedy first-token outputs; it does not manufacture calibrated probabilities.

The 30-row heads use fixed C=1, row-centered option logits, train-only StandardScaler and multinomial L2 logistic regression. Their posterior probabilities are **uncalibrated**: 30 sampled rows do not permit a valid three-class connected-group OOF calibration split. The 1,024-token head reaches DE 0.649891 / FR 0.607866 / IT 0.785680, cross-lingual 0.683547. Its ECE is 0.099514, Brier 0.434337 and NLL 0.809435. Full metrics for every run are committed, including negative and interrupted experiments.

Expanding the raw context from 1,024 to 4,096 tokens raises Macro-F1 by only 0.001857 while mean latency approximately doubles. The descriptive connected-group bootstrap interval is [0,0.002514], based on only **two independent validation components**; it is not strong evidence of statistical significance. The trained 30-row head gains 0.053877 with longer context, so both caps remain candidates for full training.

The selected hidden4096 head has numeric-class F1 **0.965116 / 0.989899 / 0.956044**, DE/FR/IT Macro-F1 **0.952097 / 0.989708 / 0.964519**, and cross-lingual Macro-F1 **0.963091**. Mean input is 1,996.15 tokens; offline cached GPU encoding plus separately timed CPU head averages 0.323438 seconds. This is not live CPU or integrated GPU-service latency. Hidden1024 reaches 0.958790 at 983.69 tokens and 0.156550 seconds, providing a measured faster alternative.

| 902-row head | OOF-selected C | Temperature | Validation ECE | Brier | NLL |
|---|---:|---:|---:|---:|---:|
| option1024 | 0.1 | 0.601435 | 0.113805 | 0.394591 | 0.728656 |
| option4096 | 1 | 1.242969 | 0.068728 | 0.351110 | 0.622029 |
| hidden1024 | 1 | 1.537363 | 0.019753 | 0.048737 | 0.084635 |
| hidden4096 | 1 | 1.415300 | 0.020791 | 0.037065 | 0.057980 |

Calibration is not uniformly beneficial: option1024 temperature worsens validation ECE from 0.051530 to 0.113805 and NLL from 0.656278 to 0.728656. Option4096 ECE also worsens, while NLL slightly improves. For the selected hidden4096 head, ECE/Brier/NLL improve from 0.022796/0.039520/0.067250 to 0.020791/0.037065/0.057980. Temperature changes probabilities, not argmax or Macro-F1. Two training groups are a material limitation; no validation recalibration is performed.

Full-booklet baseline A and annotated-minimal-gold baseline B are **unrun**: the available source contains supplied references and no gold passage labels, and the premise scope is unresolved. The source references must not be renamed gold evidence. BM25/diversification are implemented and functionally tested, but unscored as retrieval architectures. Dense/hybrid retrieval, decomposition, fine-tuning and adaptive routing are unrun; retain no unsupported performance claims.

## 6. Compute cost and lifecycle

Hard Vast budget: USD 10, with USD 2 recovery reserve. `experiments/budget.json` is the authoritative timestamped ledger. **All three task-created RTX 4090 leases were destroyed and absence verified through the paginated v1 API.** The account instance list was empty at 2026-10-07 05:08:43 UTC. Observed total account credit reduction was **USD 0.637215**, leaving **USD 9.362785**; billing can settle asynchronously. The full cache and integrity proof were pushed in commit `c30e308609846289bf1711482526513b91c45d40` before destroying the third lease. CPU heads incurred no further Vast GPU spend.

The first lease failed during provisioning when S3 logs were blocked; no GPU score is claimed. The second completed both original Apertus baselines. A long Docker-log base64 line was truncated; its archive checksum failed and that archive was rejected. Twelve JSON/JSONL results were recovered via stopped-instance UTF-8 `cat`, checked for valid JSON, exact validation IDs and source fingerprints, and metrics recomputed. `vast-provision-v2/artifact_integrity.json` records these checks honestly, without claiming the rejected archive hash passed.

The new feature transport successfully recovered the 17,939,726-byte original archive in 60 short-line parts, verifying source-side per-part and whole-archive hashes. Matrix IDs, labels, dimensions, finiteness and dataset/prompt fingerprints passed. The third lease experienced slow/interrupted downloads: two same-lease resumes preserved partial shards, and inference began only after all original shard hashes passed. Negative provisioning evidence is retained in `vast-provision-v3/events.jsonl`. The transport writes short-line base64 parts with source-side per-part and whole-archive SHA-256 checks. Results are persisted to GitHub before instance destruction. Temporary stopping is used only for recovery and never treated as cleanup. Authentication is securely bound to the Vast HTTPS destination and never copied into GPU instances or Git.

## 7. Reproducibility and application validation

`make run` starts the complete CPU frozen Apertus Docker application, including hash-locked inference dependencies and automatic pinned weight download into a named Docker volume. It needs 32 GB RAM and roughly 20 GB disk. `make run RUNTIME=workbench` starts the lightweight endpoint/retrieval mode. The Python image and PDF dependency are pinned, package hashes and TLS verified. The cloud Docker helper handles trusted proxy CA configuration without hard-coded proxy addresses or committed credentials. The Docker app processes JSON/text/PDF, returns lexical evidence with exact available provenance, and supports an Apertus OpenAI-compatible endpoint with a verified official label mapping. It fails clearly when the model is unavailable; it does not invent classifications or confidence.

36 host tests and 36 Docker tests passed without skips, covering PDF/provenance, connected splitting, duplicate isolation, metric fixtures, strict outputs and a **synthetic HTTP endpoint**. A clean GitHub checkout, complete frozen Docker startup, real CPU classification and DE/FR/IT retrieval were verified. These software checks are separate from the actual GPU scores above. Task-created validation containers were removed. See `experiments/software_validation.json`.

The live frozen classifier has also executed through the Docker HTTP workbench in DE/FR/IT. It got 2/3 synthetic cases correct and failed the French numerical contradiction; these results are retained at `experiments/frozen-live-software-v1/`, not presented as an OST score.

The selected hidden4096 Docker was subsequently built and exercised through `/health`, `/benchmarks` and actual `/api/predict` in DE/FR/IT. All 36 Docker tests passed without skips. It matched 2/3 synthetic fixtures: the Italian neutral claim was predicted as training-inferred contradiction. This diagnostic is retained without refitting at `experiments/selected-docker-validation-v1/validation.json`; it is not an OST score. Initial Docker startup failed after accumulated task build cache filled the 32 GB root disk; old task images/build cache were removed, weights/results preserved, and startup/inference then passed. Validation containers were stopped and removed.

Optional real CPU inference uses hash-locked `requirements-cpu.lock` and the pinned original model downloader. Optional lightweight head training uses `requirements-head.lock`. Existing nonempty experiment directories are rejected. Dataset/model hashes and source revision are retained. Large raw data and weights are downloaded and verified separately rather than assumed to be present in a clean checkout.

## 8. Negative results and limitations

The real three-case cross-language CPU diagnostic scored 2/3, including a French-premise/Italian-claim contradiction error, at 130 mean prompt tokens and 3.663 seconds. It is synthetic and not an OST score. A long CPU OST run was interrupted by environment publication after 30 training and 19 validation rows; partial predictions are preserved, and no full-run score is reported.

Official event guide probes returned HTTP 403; the user-supplied app.notion.com guide was blocked by current network policy, while its www.notion.so shell exposed no readable rules. The domain addition is saved in the configuration draft, not applied. The user mentioned Devpost MCP, but no Devpost capability is exposed in this session. Booklet PDF probes returned HTTP 503. The dataset card does not independently define official numeric semantics, exact submission schema, admissible premise scope or gold evidence. These block an assertion of challenge compliance, full/gold comparison and evidence Recall@k. The model source is public and weights verified; secure Vast authentication and API/S3 transport now work. SSH is unavailable, but is unnecessary for the working API recovery path.

The training split contains only two connected components (838/64 rows); validation also contains two. Reported row-level scores are descriptive. Duplicate/translation sensitivity, calibration reliability and final-test generalization require careful assessment. OCR is not implemented. The complete default Docker runtime provides actual local inference; the lightweight endpoint mode requires a configured model. No full-booklet accuracy, gold retrieval score, official class-name verification or official final submission is claimed.

## References and licensing

- [OST dataset](https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets), revision and hashes above; metadata MIT.
- [Apertus original 8B](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509), immutable revision above; Apache-2.0.
- [Official template](https://github.com/HackApertus/project-template), revision `7f2382275461baf3fa6c8855d157d86abffe9f0e`.
- [User-supplied official guide](https://app.notion.com/p/Getting-Started-Guide-Online-2975d05b692c82b1916281a56f7b6600), rules not readable in this runtime.
- [Event guide alias](https://hackapertus.notion.site/getting-started-guide-onlinehack), inaccessible from this runtime.

Code Apache-2.0; report CC-BY-4.0 following the template; pypdf BSD-3-Clause. Event-specific terms need verification before submission.
