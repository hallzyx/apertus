# Apertus Evidence Lab — English demo

[Watch or download the English video](apertus-demo.mp4?raw=true) · [Step-by-step walkthrough with screenshots](STEP_BY_STEP.md) · [PC setup](../PROBAR_EN_PC.md)

Real Chromium recording: official PDF → claim → Apertus v1.5-8B with the frozen decision head and hybrid 8k context → label, probabilities, source passages/pages, tokens and measured time → JSON download. The interface is English; claims and quoted source text retain their original languages.

Three predictions were checked first, then repeated with recording while the same native GPU backend remained loaded. Both passes reproduced labels 0, 0 and 2. These unannotated functional examples are not an accuracy benchmark. No training or internal 310-example test access occurred.

The video lasts 46.08 seconds at 1440×1100, without audio. It is an H.264 conversion of the original recording, without cuts or speed changes.

| Claim language | Label | Context tokens | Recorded inference time |
|---|---:|---:|---:|
| de | 0 | 8072 | 2.271 s |
| fr | 0 | 8149 | 2.253 s |
| it | 2 | 8161 | 2.256 s |

0 = Entailment; 1 = Neutral; 2 = Contradiction. Recorded times include retrieval and inference on the prepared GPU and are not guarantees for other machines. Source passages support inspection; the output is document-grounded inference, not political advice or independent fact checking.

The temporary GPU was destroyed after artifact recovery and verification. This recording does not publish a persistent model server. On your PC, PDF inspection works without the backend; classification requires a reachable native backend configured as documented.

## Reproducible artifacts

- [Six native predictions and session provenance](../../track_2a/experiments/apertus-live-demo-en-v1/verification.json).
- [Original browser recording](../../track_2a/experiments/apertus-live-demo-en-v1/demo.webm).
- [Video hashes and encoding](video-integrity.json).
- [GPU cleanup and account observation](resource-verification.json).
- [Browser demo runner](../../track_2a/scripts/demo_v15_browser.py).
- [Official booklet and attribution](../../track_2a/data/demo/README.md).
