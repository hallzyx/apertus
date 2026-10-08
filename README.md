# Apertus Evidence Lab · Hack Apertus Track 2A OST

Production input is **voting booklet + claim**, with DE/FR/IT and cross-lingual
NLI. The app extracts PDF passages with source pages and hashes, supplies the
validation-selected **hybrid-8k** context, and classifies using **frozen Apertus v1.5**
plus a training-only calibrated decision head. Hybrid retrieval uses supporting E5/BM25.

```bash
export LLM_NAME=swiss-ai/Apertus-v1.5-8B
export LLM_BASE_URL=https://YOUR_AUTHORIZED_APERTUS_ENDPOINT/v1
# Configure LLM_API_KEY securely if your endpoint requires authentication.
make run                    # Lightweight Docker PDF/HTTP app on port 8000
make test
```

The actual endpoint is required for classification. Without it, PDF processing
and real multilingual retrieval work; no label or probabilities are invented.
Official v1.5 weights require approved Hugging Face access for the initial download;
cached inference needs no download token. The selected native backend is required:
a generic base chat endpoint does not implement the learned hidden decision head.

On the same 276-example development validation, the best original base configuration
reached **0.671612 Macro-F1**; the final **hybrid-8k** system reaches
**0.949069**, cross-language **0.941796**,
using **8060 mean prompt tokens**. Real wrong-booklet
controls, cross-language PDF CLI calls and frontend inference are archived.
These are architecture-development results, not independent hidden-test proof.
Apertus weights remain frozen: **no LoRA or QLoRA** was trained.

The historical original full-context head scored **0.899920 on the consumed
310-example internal holdout**. That score does not apply to the newly selected
compact head. Phase2 never reopened or re-inferred those310 examples.

**The earlier 0.970353 Macro-F1 was a reference-only experiment with Apertus
2509, not v1.5 or full-booklet production.** It remains documented as historical
research and must not be presented as the final challenge score.

See [runtime and CLI](track_2a/README.md), [technical report](track_2a/technical_report.md),
and [adopted contract](track_2a/docs/challenge_contract.md). Full source booklets and
large assets are downloaded/generated separately; `track_2a/data/` contains only
small tracked examples. No Devpost integration is needed for this track.
