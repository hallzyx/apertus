# Apertus Evidence Lab — Track 2A OST technical report

## 1. Summary

The adopted challenge contract requires booklet + claim, fixed labels 0 entailment /
1 neutral / 2 contradiction, Apertus v1.5, and transparent source passages. Production
must not require reference strings. The app now implements PDF processing,
multilingual semantic/BM25 fusion retrieval, source provenance, a Docker UI and
single/batch CLI. Exact CLI/evidence schemas are project choices, not blockers.

**No full-booklet Apertus v1.5 NLI score is claimed.** Official v1.5 model access
returns HTTP 401 without approved Hugging Face authentication. The earlier 0.970353
result used Apertus 2509 and supplied references; it is historical inference/head
research, not the final challenge score or evidence of v1.5 performance.

## 2. Architecture

PDF → per-page text → overlapping source passages (1,200 characters, overlap 200)
→ multilingual E5 dense retrieval + positive-score BM25 rank fusion → five passages
→ configured Apertus v1.5 endpoint → strictly parsed numeric relation → evidence
and runtime/token metadata. Passage text remains an exact substring of extracted
page text, with page/offsets and original PDF SHA-256. Production uses no labels,
reference strings or dataset baseline scores.

The first extraction used PDF layout mode and fragmented words. Five training-only
extraction comparisons motivated plain extraction; mean full-booklet reference
5-gram coverage improved from 0.2834 to 0.8280 on validation. Both measurements are
retained; matching is approximate and not organizer evidence grading. OCR is not
implemented. Full context is explicitly byte-capped/truncation-reported when used.

## 3. Use of Apertus

Required model: `swiss-ai/Apertus-v1.5-8B`, pinned metadata revision
`a411d838600baf0e3635a3daf66fb7c55fc97bb6`, architecture
`Apertus1p5ForConditionalGeneration`, model type `apertus1p5`. It is distinct from
legacy Apertus 2509; old hidden-head parameters cannot be transferred unchanged.
The current production interface uses an authorized v1.5 endpoint through
LLM_NAME/LLM_BASE_URL/LLM_API_KEY. Model names must identify the required generation;
actual server model/revision should be verified when endpoint access becomes available.

Supporting model: `intfloat/multilingual-e5-small`, immutable revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`, MIT, CPU, original FP32 weights with
source LFS SHA-256 verification. Masked mean pooling + L2 normalization, `query:` /
`passage:` prefixes, 512-token embedding cap, 384 dimensions. It performs retrieval,
never the NLI decision. No substitute central NLI model is used.

## 4. Data

OSTswiss/MNLIoverSwissVotingBooklets, revision
`9ff08597fb79dc68cbb3af9eb1388f34d21223e6`, original JSONL SHA-256
`87ee4afd9c0ee8861106c50f05884481402ca668aecf927df68fc52d915b02fd`.
1,488 examples, DE/FR/IT, all nine document–claim language pairs. All 60 linked
source PDFs downloaded and parsed; no rows missing from 902 train / 276 validation /
310 internal holdout. PDF manifests record original URLs, hashes, pages and byte sizes.
Raw PDFs/data/weights are separately downloaded and ignored by Git.

Partitions group voting dates and connect dates sharing normalized identical claims,
without label-based partition search. Only two connected train components and two
validation components remain; this limits generalization/calibration confidence.
Reference strings are retained exclusively for oracle/alignment diagnostics.
The 310 holdout is internal, not the organizer's hidden evaluation.

## 5. Evaluation

Three-class Macro-F1 and per-class/language/pair/cross-language metrics are implemented;
actual endpoint token usage and time are recorded. Probabilities/calibration are null
when the endpoint supplies only a hard class. No fake posterior or token estimate.

Real retrieval-only validation on 276 full-booklet inputs, k=5:

| Retrieval | Mean reference 5-gram coverage | Cross-language coverage |
|---|---:|---:|
| BM25 | 0.147356 | 0.080880 |
| Multilingual E5 dense | 0.200044 | 0.191805 |
| Hybrid | 0.213526 | 0.187803 |

Hybrid k=5 was frozen from validation before one internal holdout retrieval comparison.
On the 310-row internal holdout, BM25/dense/hybrid mean reference overlap is
0.194468/0.201401/0.253554; cross-language overlap is
0.116697/0.176814/0.203536. No holdout NLI Macro-F1 has been measured.

This diagnostic measures overlap with available references, not semantic evidence
correctness, official evidence score, or NLI Macro-F1. Dense slightly exceeds hybrid
on cross-language overlap; hybrid wins aggregate overlap. NLI superiority is unproven.

| Setup | Macro-F1 | Context tokens | Latency | Evidence |
|---|---:|---:|---:|---|
| Full-booklet Apertus v1.5 | Unrun | Unmeasured | Unmeasured | Full/capped source text |
| Oracle/reference Apertus v1.5 | Unrun | Unmeasured | Unmeasured | Dataset reference, diagnostic only |
| BM25 → Apertus v1.5 | Unrun | Unmeasured | Unmeasured | Real retrieval implemented/measured separately |
| Hybrid → Apertus v1.5 | Unrun | Unmeasured | Unmeasured | Real retrieval implemented/measured separately |
| Legacy 2509 raw reference, 4096 cap | 0.453510 | 1996.15 | 0.328038 s | Supplied reference |
| Legacy 2509 trained hidden head, 4096 cap | 0.970353 | 1996.15 | 0.323438 s | Supplied reference |

Legacy timings are offline GPU encoding plus separately timed CPU head, not live CPU
or integrated service latency. Legacy exact-deduplicated score is 0.965299 on 207 rows.
Training head/calibration used train only. These old results must not be renamed
v1.5 or full-booklet scores. Historical details/negative results are preserved in
`docs/legacy_apertus_2509_report.md` and experiment directories.

## 6. Limitations

Required v1.5 model/endpoint access is missing; production NLI readiness and final
F1 therefore remain unverified. The source reference is not always a uniquely
aligned minimal passage, and approximate overlap loses typography/word-order matches.
Retrieved passages are transparent candidates, not proof of model attribution or
correct evidence. Small independent group counts, duplicated examples, translated
near-duplicates and reference-to-whole-booklet distribution shift limit conclusions.
No OCR, dense-model fine-tuning, v1.5 head training/LoRA or official hidden evaluation
has been performed. No unspecified interface/hardware limits are invented.

## 7. Reproducibility

Clean-checkout `make run` builds Dockerfile.hybrid, installs hash-locked dependencies,
downloads/verifies public retrieval weights and starts port 8000. Classification needs
an authorized v1.5 endpoint; unavailable inference produces an explicit failure.
Project-defined CLI: `python -m ost_nli predict BOOKLET.pdf CLAIM --context hybrid`.
Batch JSONL contains id/document/claim only. Source passages carry pages/offsets/hash.
Documented interface is separable from model/retrieval implementation.

Host tests pass (36 tests). Docker/PDF/retrieval validation is recorded separately
as it completes; synthetic HTTP fixture tests do not constitute real v1.5 inference.
No task GPU was rented for PDF/retrieval work. Prior three task-created leases were
destroyed and paginated API absence verified. Prior observed total credit reduction:
USD 0.637215 (asynchronous billing), out of USD 10 with USD 2 recovery reserve.

## 8. Next steps

Authenticate approved v1.5 weights or configure a verified v1.5 endpoint. Verify
native architecture/serving support before renting. Run matched full/capped, oracle,
BM25/dense/hybrid v1.5 comparisons on validation; train any new v1.5 head on train only,
freeze choices, then evaluate the 310 internal holdout once for NLI. Measure evidence
quality manually alongside reference-overlap diagnostics. No Devpost access or exact
organizer CLI signature is needed. Submission itself remains a separate action.

## 9. License

Code Apache-2.0; report CC-BY-4.0 following the template. OST dataset metadata MIT;
multilingual E5 MIT; Apertus v1.5 Apache-2.0 with published access/AUP conditions;
pypdf BSD-3-Clause. Original booklet content remains attributed to the Federal Chancellery.

## 10. References

- Adopted user-provided OST contract: `docs/challenge_contract.md`.
- https://github.com/HackApertus/project-template/tree/main/track_2a
- https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets
- https://www.bk.admin.ch/de/sammlung-der-abstimmungsbuechlein-seit-1978
- https://huggingface.co/swiss-ai/Apertus-v1.5-8B
- https://huggingface.co/intfloat/multilingual-e5-small
- Submission: http://hackapertus.ch/online-hack/submissions
