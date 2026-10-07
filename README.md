# Apertus Evidence Lab · Hack Apertus Track 2A OST

Production input is **voting booklet + claim**, with DE/FR/IT and cross-lingual
retrieval. The app extracts PDF passages with source pages and hashes, retrieves
with public multilingual E5 + BM25, and delegates NLI to **Apertus v1.5**.

```bash
export LLM_NAME=swiss-ai/Apertus-v1.5-8B
export LLM_BASE_URL=https://YOUR_AUTHORIZED_APERTUS_ENDPOINT/v1
# Configure LLM_API_KEY securely if your endpoint requires authentication.
make run                    # Docker app on port 8000; downloads verified retrieval weights
make test
```

The actual endpoint is required for classification. Without it, PDF processing
and real multilingual retrieval work; no label or probabilities are invented.
Official v1.5 weights require approved Hugging Face access and authentication;
the task obtained authorized weights and verified real native v1.5 CPU inference.
Cached weights need no Hugging Face token. Three real cross-language training PDF
cases passed through the public CLI; these are integration checks. No v1.5 NLI
Macro-F1 is claimed until the full benchmark completes.

**The earlier 0.970353 Macro-F1 was a reference-only experiment with Apertus
2509, not v1.5 or full-booklet production.** It remains documented as historical
research and must not be presented as the final challenge score.

See [runtime and CLI](track_2a/README.md), [technical report](track_2a/technical_report.md),
and [adopted contract](track_2a/docs/challenge_contract.md). Full source booklets and
large assets are downloaded/generated separately; `track_2a/data/` contains only
small tracked examples. No Devpost integration is needed for this track.
