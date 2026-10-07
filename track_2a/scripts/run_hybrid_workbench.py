"""Start the selected context workflow; download supporting weights only if needed."""
import os
import sys
if os.environ.get('SSL_CERT_FILE'):
    os.environ.setdefault('REQUESTS_CA_BUNDLE', os.environ['SSL_CERT_FILE'])
if len(sys.argv)>1:
    from ost_nli.cli import main
    main()
else:
    from ost_nli import web
    from ost_nli.runtime import production_context, embedding_retriever
    if production_context() in ('dense', 'hybrid'):
        web.RETRIEVER = embedding_retriever()
    web.serve('0.0.0.0',8000)
