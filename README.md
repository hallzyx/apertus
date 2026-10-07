# Apertus Evidence Lab · Hack Apertus Track 2A OST

Reproducible multilingual NLI with original frozen Apertus 8B and a small learned
numeric decision head over its final hidden representation. Internal strict validation: **Macro-F1 0.970353**, compared
with **0.453510** for raw option scoring at the same 4,096-token cap.
Official challenge schema, class names and premise scope still need verification.

```bash
make run                      # Complete real CPU Apertus Docker app, port 8000
make run RUNTIME=workbench     # Lightweight retrieval / authorized endpoint mode
make test                     # Unit/integration tests
```

The complete runtime needs Docker/Make, 32 GB RAM and approximately 20 GB free disk.
It downloads 16.1 GB of original weights at a pinned revision, verifies official
SHA-256 hashes, and caches them in Docker volume `apertus-models`. No inference API
key or host Python installation is required. Startup waits for the real model.
The first download can take substantial time. Weights and raw data are not committed.

See [track README](track_2a/README.md), [technical report](track_2a/technical_report.md)
and [experiment protocol](track_2a/docs/experiment_protocol.md). The preserved
`track_2a/` layout follows the official template. All metrics here are internal
supplied-reference validation; they are not official challenge or final-test scores.
