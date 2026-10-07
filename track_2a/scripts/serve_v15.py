"""Single-device, loopback-only Apertus v1.5 chat endpoint for the public CLI."""
import argparse
import json


from ost_nli.native_service import make_server


def main():
    from ost_nli.v15 import Engine
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--head')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--method',choices=['score','prompt','head'],default='score')
    p.add_argument('--port',type=int,default=8001)
    p.add_argument('--retriever-dir',help='Verified pinned E5 directory for a hybrid context-matched head')
    a=p.parse_args();engine=Engine(a.model_dir,a.device)
    head=json.loads(open(a.head).read()) if a.head else None
    pipeline=None
    if a.method=='head':
        from ost_nli.context_budget import DocumentPipeline
        retriever=None
        if head['context'].startswith('hybrid-'):
            from ost_nli.dense import MultilingualRetriever
            if not a.retriever_dir:raise ValueError('Hybrid head requires --retriever-dir')
            retriever=MultilingualRetriever(a.retriever_dir)
        pipeline=DocumentPipeline(engine,head,retriever)
    server=make_server(engine,a.method,head,a.port,pipeline)
    print('V15_SERVING_READY',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
