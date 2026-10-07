"""Prepare only public retrieval weights; Apertus v1.5 comes from configured endpoint."""
import os
import sys
from pathlib import Path
if os.environ.get('SSL_CERT_FILE'):
    os.environ.setdefault('REQUESTS_CA_BUNDLE', os.environ['SSL_CERT_FILE'])
from ost_nli.dense import download, MultilingualRetriever

directory = os.environ.get('EMBEDDING_MODEL_DIR','/models/multilingual-e5-small')
if not (Path(directory)/'verified_manifest.json').exists(): download(directory)
if len(sys.argv)>1:
    from ost_nli.cli import main
    main()
else:
    from ost_nli import web
    web.RETRIEVER = MultilingualRetriever(directory)
    web.serve('0.0.0.0',8000)
