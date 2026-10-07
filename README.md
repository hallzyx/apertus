# Apertus Evidence Lab · Hack Apertus Track 2A OST

Research implementation for multilingual NLI over Swiss voting booklets.
The official track directory is preserved as `track_2a/`.

**Status:** evaluation harness, pinned official data adapter, leakage-aware splits,
lexical retrieval and Docker workbench implemented. Real Apertus accuracy and the
final OST submission are **not yet validated**. Secure GPU access and the full
challenge specification are required to finish the research.

```bash
make run        # Docker workbench, port 8000
make test       # Python 3.11+ unit/integration tests
```

See [track README](track_2a/README.md),
[technical report](track_2a/technical_report.md) and
[experiment protocol](track_2a/docs/experiment_protocol.md).

No API keys or model weights are committed. Statistical/synthetic checks must not
be reported as Apertus challenge performance.
