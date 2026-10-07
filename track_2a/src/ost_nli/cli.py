import argparse
import json
import os
import sys
import time
from pathlib import Path
from .data import fingerprint, inspect_dataset, load_dataset, load_document, read_jsonl, split_rows
from .experiments import append_event, finish_record, new_record
from .metrics import evaluate
from .model import ApertusClient, label_map, predict
from .retrieval import retrieve


def emit(value, path=None):
    text = json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text,encoding="utf-8")
    else:
        print(text,end="")


def model_options(p):
    p.add_argument("--context",choices=["full","gold","reference","bm25","dense","hybrid"],default="bm25")
    p.add_argument("--method",choices=["prompt","constrained"],default="prompt")
    p.add_argument("--k",type=int,default=5)
    p.add_argument("--max-context-bytes",type=int,default=48000)
    p.add_argument("--diversify",action="store_true")


def main():
    parser = argparse.ArgumentParser(description="Apertus OST multilingual NLI research harness")
    subs = parser.add_subparsers(dest="command",required=True)
    p = subs.add_parser("inspect");p.add_argument("dataset");p.add_argument("--output")
    p = subs.add_parser("prepare-ost");p.add_argument("--output",required=True);p.add_argument("--source",default="data/private/official/v1.1.jsonl")
    p = subs.add_parser("majority-baseline");p.add_argument("train");p.add_argument("validation");p.add_argument("--output-dir",required=True);p.add_argument("--registry",default="experiments/registry.jsonl");p.add_argument("--id",default="cpu-majority-v1")
    p = subs.add_parser("budget-plan");p.add_argument("--ledger",default="experiments/budget.json");p.add_argument("--gpu-hourly",required=True,type=float);p.add_argument("--storage-monthly-gb",required=True,type=float);p.add_argument("--disk-gb",required=True,type=float);p.add_argument("--hours",required=True,type=float);p.add_argument("--transfer",type=float,default=0);p.add_argument("--margin",type=float,default=1.3)
    p = subs.add_parser("split");p.add_argument("dataset");p.add_argument("--output-dir",required=True);p.add_argument("--seed",type=int,default=42);p.add_argument("--validation-fraction",type=float,default=.2);p.add_argument("--test-fraction",type=float,default=.2)
    p = subs.add_parser("evaluate");p.add_argument("dataset");p.add_argument("predictions");p.add_argument("--output")
    p = subs.add_parser("retrieve");p.add_argument("booklet");p.add_argument("claim");p.add_argument("--k",type=int,default=5);p.add_argument("--diversify",action="store_true")
    p = subs.add_parser("predict-frozen");p.add_argument("booklet");p.add_argument("claim");p.add_argument("--model-dir",required=True);p.add_argument("--head-dir",required=True);p.add_argument("--device",choices=["cpu","cuda"],default="cpu");p.add_argument("--context",choices=["full","bm25"],default="full");p.add_argument("--k",type=int,default=5)
    p = subs.add_parser("predict");p.add_argument("booklet");p.add_argument("claim");model_options(p);p.set_defaults(context='hybrid')
    p = subs.add_parser("predict-batch");p.add_argument("input");p.add_argument("--output",required=True);model_options(p);p.set_defaults(context='hybrid')
    p = subs.add_parser("experiment");p.add_argument("dataset");p.add_argument("--id",required=True);p.add_argument("--split-name",required=True,choices=["train","validation","test"]);p.add_argument("--output-dir",required=True);p.add_argument("--registry",default="experiments/registry.jsonl");p.add_argument("--estimated-cost",type=float,default=0);p.add_argument("--gpu",default="unknown");p.add_argument("--notes",default="");model_options(p)
    p = subs.add_parser("serve");p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8000)
    subs.add_parser("self-test")
    args = parser.parse_args()
    try:
        if args.command == "budget-plan":
            from .budget import estimate
            emit(estimate(args.ledger,args.gpu_hourly,args.storage_monthly_gb,args.disk_gb,args.hours,args.transfer,args.margin))
        elif args.command == "majority-baseline":
            from .baselines import training_majority
            train, rows = load_dataset(args.train),load_dataset(args.validation)
            out = Path(args.output_dir)
            if out.exists() and any(out.iterdir()):
                raise ValueError("Existing baseline output directory cannot be overwritten")
            args.split_name="validation";args.model_name="training-majority-prior";args.context="none";args.k=0;args.diversify=False;args.max_context_bytes=0;args.method="argmax_training_prior";args.mapping=None;args.gpu="CPU";args.estimated_cost=0.;args.notes="Non-Apertus statistical floor for evaluator validation; not a challenge submission or semantic NLI model"
            record = new_record(args.id,rows,args)
            record["training_dataset_sha256"]=fingerprint(train)
            append_event(args.registry,record)
            start=time.perf_counter()
            predictions=training_majority(train,rows)
            metrics=evaluate(rows,predictions)
            final=finish_record(record,metrics,time.perf_counter()-start)
            out.mkdir(parents=True,exist_ok=True)
            emit({"dataset_sha256":fingerprint(rows),"configuration":record,"predictions":predictions},out/"predictions.json")
            emit(metrics,out/"metrics.json");emit(final,out/"experiment.json")
            append_event(args.registry,final)
            emit(metrics)
        elif args.command == "prepare-ost":
            from .official import prepare
            emit(prepare(args.source,args.output))
        elif args.command == "inspect":
            emit(inspect_dataset(load_dataset(args.dataset)), args.output)
        elif args.command == "split":
            rows = load_dataset(args.dataset)
            partitions = split_rows(rows,args.seed,args.validation_fraction,args.test_fraction)
            out = Path(args.output_dir)
            if out.exists() and any(out.iterdir()):
                raise ValueError("Split output directory must be empty; never overwrite a frozen split")
            out.mkdir(parents=True,exist_ok=True)
            for name, subset in partitions.items():
                (out/(name+".jsonl")).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in subset),encoding="utf-8")
            manifest = {"dataset_sha256":fingerprint(rows),"seed":args.seed,"unit":"connected_booklet_and_normalized_duplicate_claim_groups","requested_fractions":{"validation":args.validation_fraction,"test":args.test_fraction},"algorithm":"Greedy target-deficit allocation, largest component first, seeded hash tie break; no label-based optimization","partitions":{name:{"sha256":fingerprint(rs),"example_ids":[r["id"] for r in rs],"booklet_ids":sorted({r["booklet_id"] for r in rs}),"distribution":inspect_dataset(rs)} for name,rs in partitions.items()}}
            emit(manifest,out/"manifest.json")
            emit({"manifest":str(out/"manifest.json"),"sizes":{name:len(rs) for name,rs in partitions.items()}})
        elif args.command == "evaluate":
            rows = load_dataset(args.dataset)
            bundle = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
            if bundle.get("dataset_sha256") != fingerprint(rows):
                raise ValueError("Prediction dataset fingerprint mismatch")
            emit(evaluate(rows,bundle["predictions"]),args.output)
        elif args.command == "retrieve":
            emit(retrieve(load_document(args.booklet)["passages"],args.claim,args.k,args.diversify))
        elif args.command == "predict-frozen":
            from .frozen import FrozenApertus
            mapping = label_map() if os.environ.get("OST_LABEL_MAP") else None
            emit(FrozenApertus(args.model_dir,args.head_dir,args.device).predict(
                 load_document(args.booklet),args.claim,args.context,args.k,mapping))
        elif args.command == "predict":
            if args.context in ('gold','reference'):
                raise ValueError("Reference/gold evidence is an evaluation diagnostic, never production input")
            os.environ['APERTUS_REQUIRED_GENERATION']='v1.5'
            doc = load_document(args.booklet)
            began = time.perf_counter()
            mode = args.context
            if mode in ('dense','hybrid'):
                from .dense import MultilingualRetriever
                retriever = MultilingualRetriever(os.environ.get('EMBEDDING_MODEL_DIR','/models/multilingual-e5-small'))
                doc = {**doc,'passages':retriever.retrieve(doc['passages'],args.claim,args.k,mode)}
                mode = 'full'
            result = predict({**doc,"claim":args.claim},ApertusClient(),label_map(),mode,args.k,args.max_context_bytes,args.method,args.diversify)
            result['retrieval'] = args.context
            result['model_inference_time_ms'] = result['inference_time_ms']
            result['inference_time_ms'] = (time.perf_counter()-began)*1000
            result['latency_seconds'] = result['inference_time_ms']/1000
            emit(result)
        elif args.command == 'predict-batch':
            os.environ['APERTUS_REQUIRED_GENERATION']='v1.5'
            if args.context in ('gold','reference'):
                raise ValueError('Production batch input is booklet + claim, never reference/gold evidence')
            if Path(args.output).exists(): raise ValueError('Existing batch output cannot be overwritten')
            client = ApertusClient(); documents = {}; results = []; seen = set(); retriever = None
            if args.context in ('dense','hybrid'):
                from .dense import MultilingualRetriever
                retriever = MultilingualRetriever(os.environ.get('EMBEDDING_MODEL_DIR','/models/multilingual-e5-small'))
            for row in read_jsonl(args.input):
                if not isinstance(row.get('id'),str) or not row['id'] or row['id'] in seen:
                    raise ValueError('Batch IDs must be unique nonempty strings')
                seen.add(row['id'])
                path = Path(args.input).parent / row['document']
                if str(path) not in documents: documents[str(path)] = load_document(path)
                doc = documents[str(path)]; began = time.perf_counter(); mode = args.context
                if retriever:
                    doc = {**doc,'passages':retriever.retrieve(doc['passages'],row['claim'],args.k,mode)}
                    mode = 'full'
                result = predict({**doc,'id':row['id'],'claim':row['claim']},client,label_map(),mode,args.k,args.max_context_bytes,args.method,args.diversify)
                result['retrieval']=args.context;result['model_inference_time_ms']=result['inference_time_ms']
                result['inference_time_ms']=(time.perf_counter()-began)*1000;result['latency_seconds']=result['inference_time_ms']/1000
                results.append(result)
            if not results: raise ValueError('Empty batch')
            emit({'predictions':results,'production_input':'booklet+claim only'},args.output)
        elif args.command == "experiment":
            if args.estimated_cost < 0:
                raise ValueError("Cost cannot be negative")
            rows = load_dataset(args.dataset)
            if args.context == "full" and any(r.get("document_scope") == "provided_reference" for r in rows):
                raise ValueError("Full-booklet baseline requires full booklets; use reference mode for provided-reference data")
            out = Path(args.output_dir)
            if out.exists() and any(out.iterdir()):
                raise ValueError("Experiment output directory must be empty")
            args.mapping = label_map()
            client = ApertusClient()
            retriever = None
            if args.context in ('dense','hybrid'):
                from .dense import MultilingualRetriever, REPO, REVISION
                retriever = MultilingualRetriever(os.environ.get('EMBEDDING_MODEL_DIR','/models/multilingual-e5-small'))
            args.model_name = client.model
            record = new_record(args.id,rows,args)
            if retriever:
                record['supporting_retrieval_model']={'repo':REPO,'revision':REVISION}
            # Capture dirty status: commit SHA alone does not identify uncommitted code.
            import subprocess
            try:
                record["git_dirty"] = bool(subprocess.check_output(["git","status","--porcelain"],text=True,stderr=subprocess.DEVNULL))
            except (subprocess.CalledProcessError,FileNotFoundError):
                record["git_dirty"] = None
            append_event(args.registry,record)
            out.mkdir(parents=True,exist_ok=True)
            predictions, start = [], time.perf_counter()
            try:
                for r in rows:
                    began = time.perf_counter()
                    mode = args.context
                    prepared = r
                    if retriever:
                        prepared = {**r,'passages':retriever.retrieve(r['passages'],r['claim'],args.k,mode)}
                        mode = 'full'
                    pred = predict(prepared,client,args.mapping,mode,args.k,args.max_context_bytes,args.method,args.diversify)
                    pred['retrieval']=args.context
                    pred['model_inference_time_ms']=pred['inference_time_ms']
                    pred['latency_seconds']=time.perf_counter()-began
                    pred['inference_time_ms']=pred['latency_seconds']*1000
                    predictions.append(pred)
                    append_event(out/"partial_predictions.jsonl",pred)
                metrics = evaluate(rows,predictions)
                emit({"dataset_sha256":fingerprint(rows),"configuration":record,"predictions":predictions},out/"predictions.json")
                emit(metrics,out/"metrics.json")
                final = finish_record(record,metrics,time.perf_counter()-start)
                append_event(args.registry,final)
                emit(final,out/"experiment.json")
                emit(metrics)
            except Exception:
                failure = {**record,"status":"failed","completed_examples":len(predictions),"runtime_seconds":time.perf_counter()-start,"notes":record["notes"]+"; run failed; partial predictions retained, no complete-run score"}
                append_event(args.registry,failure)
                emit(failure,out/"experiment.json")
                raise
        elif args.command == "serve":
            from .web import serve
            serve(args.host,args.port)
        elif args.command == "self-test":
            import unittest
            suite = unittest.defaultTestLoader.discover("tests")
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            if result.testsRun == 0 or not result.wasSuccessful():
                raise SystemExit(1)
    except (ValueError,RuntimeError,OSError,KeyError) as e:
        print(f"Error: {e}",file=sys.stderr)
        raise SystemExit(2)
