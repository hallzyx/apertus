"""Use the frozen production context consistently across CLI and web."""
import json
import os
from pathlib import Path


def production_context():
    context = os.environ.get('RETRIEVAL_MODE')
    if not context:
        path = Path(__file__).resolve().parents[2]/'deployment/v15-selection.json'
        selection = json.loads(path.read_text()) if path.exists() else {}
        context = selection.get('choice', selection).get('context', 'hybrid')
    if context not in ('full', 'bm25', 'dense', 'hybrid'):
        raise ValueError('Production context must be full, bm25, dense or hybrid')
    return context
