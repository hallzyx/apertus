# Phase 2 — preliminary falsification audit

Status: local diagnostics complete; actual Apertus claim-only/wrong-document controls pending. No phase-2 architecture selected; historical production selection and consumed 310 holdout remain frozen. No LoRA.

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
