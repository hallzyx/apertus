"""Use the frozen production context consistently across CLI and web."""
import json
import os
from pathlib import Path


def selected_choice():
    path=Path(__file__).resolve().parents[2]/'deployment/v15-selection.json'
    selection=json.loads(path.read_text()) if path.exists() else {}
    return selection.get('choice',selection)


def document_service_url(strict=False):
    explicit=os.environ.get('FROZEN_BASE_URL','').rstrip('/')
    if explicit:return explicit
    if selected_choice().get('transport')=='document-service':
        url=os.environ.get('LLM_BASE_URL','').rstrip('/')
        if not url and strict:raise ValueError('Selected context-matched head requires its native document service URL')
        return url[:-3] if url.endswith('/v1') else (url or None)
    return None


def production_context():
    choice=selected_choice()
    if choice.get('transport')=='document-service':
        # Send the complete document to the backend which owns matching packing.
        return 'full'
    context = os.environ.get('RETRIEVAL_MODE')
    if not context:
        context = choice.get('context', 'hybrid')
    if context not in ('full', 'bm25', 'dense', 'hybrid'):
        raise ValueError('Production context must be full, bm25, dense or hybrid')
    return context


def embedding_retriever():
    from .dense import download, MultilingualRetriever
    directory = os.environ.get('EMBEDDING_MODEL_DIR', '/models/multilingual-e5-small')
    if not (Path(directory)/'verified_manifest.json').exists():
        download(directory)
    return MultilingualRetriever(directory)
