import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from ost_nli.data import fingerprint, inspect_dataset, load_dataset, load_document, split_rows
from ost_nli.metrics import calibration, classification, evaluate
from ost_nli.model import ApertusClient, label_map, predict, select_context
from ost_nli.retrieval import retrieve


def example(i=0, group=None, language="de", document_language="de"):
    return {"id":str(i),"document_id":f"d-{group or i}-{document_language}","booklet_id":str(group or i),"claim":f"jährlicher Beitrag 100 Franken Gruppe {group or i}","label":i%3,"claim_language":language,"document_language":document_language,"evidence_ids":["p1"],"passages":[{"id":"p1","page":1,"text":"Jährlicher Beitrag 100 Franken."},{"id":"p2","page":2,"text":"Andere Vorlage betrifft die Schule."}]}


class MetricsTests(unittest.TestCase):
    def test_known_confusion_matrix(self):
        result = classification([0,0,1,1,2,2],[0,1,1,2,2,0])
        self.assertEqual(result["macro_f1"],.5)
        self.assertEqual(result["confusion_matrix"],[[1,1,0],[0,1,1],[1,0,1]])
    def test_fixed_class_macro(self):
        self.assertAlmostEqual(classification([0],[0])["macro_f1"],1/3)
    def test_empty_not_success(self):
        with self.assertRaises(ValueError):classification([],[])
    def test_invalid_label(self):
        with self.assertRaises(ValueError):classification([0],[3])
    def test_perfect_calibration(self):
        result = calibration([0,1,2],[[1,0,0],[0,1,0],[0,0,1]])
        self.assertEqual(result["brier_score"],0)
        self.assertEqual(result["ece"],0)
        self.assertEqual(result["negative_log_likelihood"],0)
    def test_uniform_calibration(self):
        result = calibration([0,1,2],[[1/3]*3]*3)
        self.assertAlmostEqual(result["brier_score"],2/3)
        self.assertAlmostEqual(result["negative_log_likelihood"],math.log(3))
        self.assertAlmostEqual(result["ece"],0)
    def test_reject_invalid_probabilities(self):
        for p in [[.1,.1,.1],[float('nan'),0,1],[-1,1,1]]:
            with self.assertRaises(ValueError):calibration([0],[p])
    def test_slices_and_evidence(self):
        rows = [example(0),example(1,language="fr"),example(2,language="it")]
        predictions = [{"id":str(i),"label":i,"context_tokens":10+i,"latency_seconds":.1,"evidence_ids":["p1"]} for i in range(3)]
        m = evaluate(rows,predictions)
        self.assertEqual(m["macro_f1"],1)
        self.assertEqual(m["cross_lingual"]["n"],2)
        self.assertEqual(m["average_context_tokens"],11)
        self.assertEqual(m["evidence"]["recall"],1)
        self.assertIsNone(m["calibration"])
    def test_missing_usage_not_zero(self):
        m = evaluate([example(0)],[{"id":"0","label":0}])
        self.assertIsNone(m["average_context_tokens"])
        self.assertEqual(m["context_tokens_coverage"],0)
        self.assertIsNone(m["evidence"]["recall"])
    def test_prediction_alignment(self):
        for ps in [[],[{"id":"other","label":0}],[{"id":"0","label":0},{"id":"0","label":0}]]:
            with self.assertRaises(ValueError):evaluate([example(0)],ps)


class DataTests(unittest.TestCase):
    def write(self, rows, path):
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    def test_group_split_translations_and_repeatability(self):
        rows = [example(i,group=f"g{i//3}",language=["de","fr","it"][i%3]) for i in range(30)]
        splits = split_rows(rows)
        group_sets = [set(r["booklet_id"] for r in rs) for rs in splits.values()]
        self.assertFalse(group_sets[0]&group_sets[1] or group_sets[0]&group_sets[2] or group_sets[1]&group_sets[2])
        reversed_splits = split_rows(rows[::-1])
        for name in splits:self.assertEqual(fingerprint(splits[name]),fingerprint(reversed_splits[name]))
    def test_small_split_not_empty(self):
        self.assertTrue(all(split_rows([example(i) for i in range(3)],validation_fraction=.1,test_fraction=.8).values()))
    def test_insufficient_groups(self):
        with self.assertRaises(ValueError):split_rows([example(0),example(1)])
    def test_repeated_claims_never_cross_partitions(self):
        rows=[example(i) for i in range(10)];rows[1]["claim"]=rows[0]["claim"]
        splits=split_rows(rows)
        membership={r["id"]:name for name,rs in splits.items() for r in rs}
        self.assertEqual(membership["0"],membership["1"])
    def test_single_duplicate_component_fails(self):
        rows=[example(i) for i in range(4)]
        for r in rows:r["claim"]="identical claim"
        with self.assertRaises(ValueError):split_rows(rows)
    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"data.jsonl";self.write([example(0),example(0)],p)
            with self.assertRaises(ValueError):load_dataset(p)
    def test_invalid_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"data.jsonl";r=example(0);r["evidence_ids"]=["missing"];self.write([r],p)
            with self.assertRaises(ValueError):load_dataset(p)
    def test_same_doc_group_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"data.jsonl";a=example(0);b=example(1);b["document_id"]=a["document_id"];self.write([a,b],p)
            with self.assertRaises(ValueError):load_dataset(p)
    def test_load_text_provenance_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"booklet.txt";p.write_text("First paragraph.\n\nSecond paragraph.")
            d=load_document(p);self.assertEqual(len(d["passages"]),2);self.assertIsNone(d["passages"][0]["page"])
    def test_load_pdf_page_and_text(self):
        import importlib.util
        if importlib.util.find_spec("pypdf") is None:self.skipTest("Install hash-locked PDF dependency or run Docker tests")
        content=b"BT /F1 12 Tf 72 720 Td (The contribution is 100 francs.) Tj ET"
        objects=[b"<< /Type /Catalog /Pages 2 0 R >>",b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",b"<< /Length "+str(len(content)).encode()+b" >>\nstream\n"+content+b"\nendstream"]
        pdf=b"%PDF-1.4\n";offsets=[0]
        for i,o in enumerate(objects,1):offsets.append(len(pdf));pdf+=str(i).encode()+b" 0 obj\n"+o+b"\nendobj\n"
        xref=len(pdf);pdf+=b"xref\n0 6\n0000000000 65535 f \n"+b"".join(f"{n:010d} 00000 n \n".encode() for n in offsets[1:])+b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"+str(xref).encode()+b"\n%%EOF"
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"booklet.pdf";p.write_bytes(pdf);d=load_document(p)
            self.assertEqual(d["passages"][0]["page"],1);self.assertIn("100 francs",d["passages"][0]["text"])
    def test_inspection(self):
        r=inspect_dataset([example(i) for i in range(3)])
        self.assertEqual(r["examples"],3);self.assertEqual(r["classes"],{"0":1,"1":1,"2":1})


class RetrievalTests(unittest.TestCase):
    def test_relevant_first_unicode(self):
        ps=retrieve(example()["passages"],"JÄHRLICHER Beitrag 100 Franken",k=1)
        self.assertEqual(ps[0]["id"],"p1")
    def test_k_bounded(self):
        self.assertEqual(len(retrieve(example()["passages"],"abc",k=99)),2)
    def test_diversification_no_duplicate(self):
        ps=retrieve(example()["passages"],"Beitrag",k=2,diversify=True)
        self.assertEqual(len({p["id"] for p in ps}),2)
    def test_gold_exact(self):
        selected,_=select_context(example(),mode="gold")
        self.assertEqual([p["id"] for p in selected],["p1"])
    def test_context_cap_multibyte(self):
        r=example();r["passages"][0]["text"]="é"*1000
        selected,text=select_context(r,mode="full",max_bytes=128)
        self.assertLessEqual(len(text.encode()),128)
        self.assertTrue(selected[0]["truncated"])
    def test_context_modes_do_not_mutate_data(self):
        r=example();before=copy.deepcopy(r)
        for mode in ["full","gold","bm25"]:select_context(r,mode=mode)
        self.assertEqual(r,before)


class ModelTests(unittest.TestCase):
    mapping = {"0":"entailment","1":"contradiction","2":"neutral"} # synthetic test convention only
    def client(self, response):
        return ApertusClient("http://localhost:8001/v1","test-Apertus",transport=lambda payload:response)
    def test_prediction(self):
        response={"choices":[{"finish_reason":"stop","message":{"content":'{"label": 1}'}}],"usage":{"prompt_tokens":123}}
        p=predict(example(),self.client(response),self.mapping)
        self.assertEqual(p["label"],1);self.assertEqual(p["context_tokens"],123)
        self.assertIsNone(p["probabilities"])
    def test_invalid_outputs_never_fallback(self):
        for content in ['{"label":4}','{"label":true}','{"label":"0"}','Some explanation','{"label":0,"other":1}']:
            with self.assertRaises(ValueError):predict(example(),self.client({"choices":[{"message":{"content":content}}]}),self.mapping)
    def test_truncated_generation_rejected(self):
        with self.assertRaises(ValueError):predict(example(),self.client({"choices":[{"finish_reason":"length","message":{"content":'{"label":0}'}}]}),self.mapping)
    def test_constrained_payload(self):
        seen=[]
        def transport(payload):
            seen.append(payload);return {"choices":[{"message":{"content":'{"label":0}'}}]}
        c=ApertusClient("http://localhost:8001/v1","test-Apertus",transport=transport)
        p=predict(example(),c,self.mapping,method="constrained")
        self.assertIn("response_format",seen[0]);self.assertIsNone(p["context_tokens"])
    def test_unverified_mapping_not_defaulted(self):
        with self.assertRaises(ValueError):label_map('{"0":"neutral"}')
    def test_endpoint_security(self):
        for url in ["http://remote.example/v1","https://key:secret@example.com/v1","file:///tmp/model"]:
            with self.assertRaises(ValueError):ApertusClient(url,"Apertus")
    def test_non_apertus_not_submission(self):
        with self.assertRaises(ValueError):ApertusClient("http://localhost:8001/v1","other-model")
