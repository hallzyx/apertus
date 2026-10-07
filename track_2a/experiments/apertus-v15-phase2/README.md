# Phase 2 — grounding controls and efficiency study

Status: actual grounding controls completed and checksum verified; registered efficiency study next. No phase-2 architecture selected; historical production selection and consumed 310 holdout remain frozen. No LoRA.

## Claim artifact probes

Train-only grouped C selection and OOF temperature calibration; vocabulary fitted inside each fold. Validation only. These CPU lexical probes are NOT the Apertus claim-only control.

| Probe | Validation Macro-F1 |
|---|---:|
| claim-word-tfidf | 0.620765 |
| claim-char-tfidf | 0.626285 |
| metadata | 0.507226 |

## Cached frozen-head context interventions

The unchanged full-context trained head is applied to previously recorded actual Apertus encodings. Zero new GPU calls, no refitting. These are sensitivity/evidence diagnostics, not a selected architecture or a context-matched training sweep. Timings are cached GPU encoding plus previously measured retrieval, excluding new CPU head overhead; not fresh end-to-end serving.

| Context | Validation Macro-F1 | Mean input tokens | Cached seconds | E/C decision preservation |
|---|---:|---:|---:|---:|
| full | 0.894461 | 14106.1 | 3.635 | 1.000 |
| bm25 | 0.661108 | 1654.6 | 0.401 | 0.454 |
| dense | 0.848070 | 1395.8 | 0.359 | 0.749 |
| hybrid | 0.861704 | 1656.6 | 0.426 | 0.781 |
| reference | 0.940951 | 2269.7 | 0.540 | 0.820 |

Oracle/reference context is diagnostic only and cannot be required in production. Decision preservation is not proof of evidence sufficiency: the full head is being applied under a context distribution shift. Five-passage hybrid looks promising, but grounding controls must pass before optimizing it.

## Dataset structure and scope risks

Validation: 276 rows, 207 unique normalized claim/document pairs, three voting events, two duplicate-connected components. Deterministic deduplicated Macro-F1: 0.883035, versus row-level 0.894461. Exploratory event-resampling 95% percentiles: [0.863617, 0.944241]; three events do not support a reliable population interval and two are duplicate-connected.

`premise-scope-risk.json` records manually inspected example ost-v1.1-0732: stored Neutral label and COVID reference, while full booklet pages 6/32 discuss climate neutrality2050 and Federal Council arguments for accelerating fossil-fuel transition. This is a possible reference-vs-booklet scope mismatch, not proof that every neutral label is wrong. Original labels remain unchanged. `final-error-analysis.jsonl` inventories all29 validation errors; five have source-inspected qualified hypotheses (scope, numeric equivalence, truncation/current-vs-proposed rules); others remain undetermined rather than receiving invented causal explanations.

## Evidence and confidence

`evidence-diagnostics.json` reports reference5gram overlap, not organizer scoring. Mean coverage: full .586, BM25 .147, dense .201, hybrid .217. Neutral references can be deliberately unrelated to claims: low overlap there is not necessarily retrieval failure. Cached hybrid E/C decision preservation .781 is a limited diagnostic, not minimal proof validation.

Original full head validation: ECE .082754 (10 bins), Brier .173079, NLL .321408. Confidence reliability is recorded in CPU audit: the .50–.70 bucket has accuracy .617; .95–1 bucket is90/90 correct on this small validation. Adaptive routing has NOT been selected or tuned yet.

## Next gate and budget

1. Actual Apertus claim-only train902/validation276 encodings and train-only head.
2. Same original full head on wrong same-language different-event documents, seeds42/1337, plus empty/generic context.
3. If scores retain>=90% of correct-context F1 or lie within .05, investigate artifacts and stop architecture optimization. Wrong-document original-label scoring measures artifact retention, not actual NLI correctness for swapped booklets.
4. Only then context/retrieval matched-head efficiency experiments, evidence, final architecture.

Vast observed balance at phase2 start: USD6.534131. New discretionary cap USD4.50; preserve USD2.00. Local audit snapshot predates GPU launch; current lease/cost status is authoritative in `../budget.json`. Quoted known RTX A6000 48GB /60GB disk: USD0.416667/h, expected1.5h (~USD0.63), conservative3h plus30GB download/recovery margin <=USD1.351. Recheck offer and account credit immediately before leasing; host quotes are not reservations.

`connected-component-resampling.json` also resamples the two duplicate-connected validation units (199/77 rows), with descriptive95% percentiles [.863617, .903166]. This is not an independent-test confidence interval. `claim-artifact-catalog.json` confirms697 unique normalized train claim/document pairs and zero exact normalized claim overlap with validation. `wrong-document-overlap-diagnostic.json` measures~.021 reference5gram coverage for both wrong-document permutations, versus~.586 for correct capped context; overlap is not swapped-pair NLI ground truth.

Frontend verification:38 application tests passed both host and minimal Docker; two separate NumPy research tests passed on host. Live Docker parsed a SHA-verified official72-page French booklet into157 passages and selected52 exact page/offset source quotes. The endpoint is unconfigured for classification on this cloud host; actual native GPU controls run on the disposable Vast lease. No live classification is claimed for the frontend.

Error comparison: the restricted base class scorer is correct on20 of29 cases that the learned full head gets wrong. Both manually inspected75%/three-quarters cases have correct base full-context Entailment but incorrect learned-head decisions. This locates a decision-mapping failure in those cases; it does not establish that Apertus weights need fine-tuning or explain why the mapping fails.

## Actual Apertus grounding controls

Registered gate: **GO_WITH_LIMITATIONS**. No warning conditions. All2006 actual native GPU forwards completed; original model weights/head unchanged, no LoRA, no consumed310 access.

| Condition | Original-label validation Macro-F1 |
|---|---:|
| correct_document_frozen_head | 0.894461 |
| claim_only_train_fitted_head | 0.699640 |
| claim-only | 0.210229 |
| wrong-42 | 0.565341 |
| wrong-1337 | 0.553328 |
| generic | 0.174688 |

Wrong-document scores retain only 63.2%/61.9% of correct-context score. These substantial drops support context dependence under the registered controls. They do not prove correct evidence reasoning: original labels are not true labels for intervened premises. Refitted claim-only capability .700 shows residual claim artifacts; empty-context original head .210 measures a different intervention. Only two wrong permutations and few events were tested.

Source encodings and fitted claim-only audit: `../apertus-v15-phase2-controls-v1`; recovered archive SHA `d023629cf7e283a66b13b05c1f92b4c05165fc50544a7eac47aa049abe437c0b`. All105 export-part SHA checks, complete archive checksum and experiment file checks passed. Task-owned instance54722568 destroyed and verified absent via paginated provider API. New context study is preregistered in `context-protocol.json`; no efficiency architecture is selected yet.

## Additional claim similarity audit

Fixed1-nearest-neighbor E5 claim-only classifier: validation Macro-F1 .583725;64 of276 nearest-training claim cosines exceed .95,12 exceed .98. Three manually inspected high-similarity pairs are near paraphrases of Assembly recommend/reject/accept statements across different voting events. Exact normalized overlap being zero therefore does not establish semantic independence. High cosine alone is not duplicate-task or leakage proof: a numerical or negation change can reverse NLI, and different booklets are genuinely different premises. These development probes are not deployed.

The registered final check includes an original full-head A6000 recheck to resolve the historical A40-versus-A6000 numerical confound. A compact candidate must additionally pass condition-matched wrong-booklet controls and three actual cross-language PDF CLI calls before deployment changes.
