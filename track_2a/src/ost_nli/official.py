"""Revision-pinned OST dataset downloader and provided-premise adapter."""
import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from .data import inspect_dataset, load_dataset, read_jsonl

REPOSITORY = "OSTswiss/MNLIoverSwissVotingBooklets"
REVISION = "9ff08597fb79dc68cbb3af9eb1388f34d21223e6"
SOURCE_SHA256 = "87ee4afd9c0ee8861106c50f05884481402ca668aecf927df68fc52d915b02fd"
URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/v1.1.jsonl"


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Output exists; do not overwrite canonical data")
    source.parent.mkdir(parents=True,exist_ok=True)
    if not source.exists():
        with urllib.request.urlopen(URL,timeout=90) as response:
            data = response.read(30_000_000)
        if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
            raise ValueError("Official dataset checksum mismatch; refusing downloaded data")
        source.write_bytes(data)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Official source checksum mismatch; expected pinned OST revision")
    raw = read_jsonl(source)
    converted = []
    claims = defaultdict(set)
    exact_examples = Counter()
    for i,r in enumerate(raw):
        ref = r["reference_string"]
        url = r["booklet_url"]
        date = r["booklet_publish_date"]
        ref_hash = hashlib.sha256(ref.encode()).hexdigest()[:20]
        if r["claim_language"] not in ("de","fr","it") or r["reference_language"] not in ("de","fr","it"):
            raise ValueError("Unexpected OST language; adapter must be reviewed")
        if type(r["entailment_label"]) is not int or r["entailment_label"] not in (0,1,2):
            raise ValueError("Unexpected official numeric label")
        parts = [p.strip() for p in re.split(r"\n\s*\n",ref) if p.strip()]
        passages = [{"id":f"{ref_hash}-p{j+1}","page":None,"text":text,"language":r["reference_language"],"source_url":url,"source_reference_sha256":hashlib.sha256(ref.encode()).hexdigest()} for j,text in enumerate(parts)]
        converted.append({"id":f"ost-v1.1-{i:04d}","document_id":f"{date}-{r['reference_language']}-{ref_hash}","booklet_id":date,"claim":r["claim"],"claim_language":r["claim_language"],"document_language":r["reference_language"],"label":r["entailment_label"],"passages":passages,"evidence_ids":[],"evidence_annotation_available":False,"provided_reference_ids":[p["id"] for p in passages],"document_scope":"provided_reference","source_booklet_url":url,"source_row_index":i,"source_revision":REVISION})
        claims[r["claim"]].add(date)
        exact_examples[(r["claim"],ref,r["claim_language"],r["reference_language"],r["entailment_label"])] += 1
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in converted),encoding="utf-8")
    canonical = load_dataset(output)
    audit = {**inspect_dataset(canonical),"source_repository":REPOSITORY,"source_revision":REVISION,"source_file_sha256":SOURCE_SHA256,"license_metadata":"MIT","source_fields":sorted(raw[0]),"original_document_urls":len({r['booklet_url'] for r in raw}),"voting_dates":len({r['booklet_publish_date'] for r in raw}),"distinct_reference_strings":len({r['reference_string'] for r in raw}),"exact_duplicate_examples":sum(n-1 for n in exact_examples.values()),"exact_claims_crossing_booklet_dates":sum(len(d)>1 for d in claims.values()),"provided_reference_annotation":"NLI premise excerpts, not annotated minimal gold evidence","numeric_class_semantics":"Not documented in source card; official rules confirmation required","baseline_score_handling":"Excluded from inputs, features and metrics; unknown source score semantics","full_booklets_available":False,"page_provenance_available":False}
    output.with_suffix(".audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return audit
