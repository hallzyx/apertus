import datetime
import json
import subprocess
from pathlib import Path
from .data import fingerprint


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def git_commit():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):
        return None


def append_event(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")


def new_record(experiment_id, rows, args):
    return {"experiment_id":experiment_id,"timestamp":utc_now(),"status":"started","git_commit":git_commit(),"dataset_split":args.split_name,"dataset_sha256":fingerprint(rows),"model":args.model_name,"retrieval_configuration":{"mode":args.context,"k":args.k,"diversify":args.diversify,"max_context_bytes":args.max_context_bytes},"prompt_configuration":{"method":args.method,"version":"ost-nli-v1","label_map":args.mapping},"training_configuration":None,"macro_f1":None,"per_class_f1":None,"per_language_performance":None,"cross_lingual_performance":None,"evidence_metrics":None,"average_context_tokens":None,"average_inference_latency":None,"gpu":args.gpu,"runtime_seconds":None,"estimated_compute_cost":args.estimated_cost,"actual_compute_cost":None,"notes":args.notes}


def finish_record(record, metrics, runtime):
    return {**record,"timestamp":utc_now(),"status":"completed","runtime_seconds":runtime,"macro_f1":metrics["macro_f1"],"per_class_f1":{k:v["f1"] for k,v in metrics["per_class"].items()},"per_language_performance":metrics["by_language"],"cross_lingual_performance":metrics["cross_lingual"],"evidence_metrics":metrics["evidence"],"average_context_tokens":metrics["average_context_tokens"],"average_inference_latency":metrics["average_latency_seconds"]}
