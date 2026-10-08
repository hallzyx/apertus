"""Nine predetermined qualitative development inspections; not organizer scoring."""
import json,hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ost_nli.data import load_dataset
from ost_nli.model import select_context
from ost_nli.documents import reference_coverage
root=Path(__file__).resolve().parents[1];phase=root/'experiments/apertus-v15-phase2'
result_path=phase/'final-results.json'
if (phase/'accuracy-results.json').exists():
 alternative=json.loads((phase/'accuracy-results.json').read_text())
 if alternative['final_choice']=='hybrid-8k' and not alternative['guard_failures']:result_path=phase/'accuracy-results.json'
final=json.loads(result_path.read_text());choice=final['final_choice']
rows={r['id']:r for r in load_dataset(root/'data/private/full-booklets-v2/validation.jsonl')}
ids=json.loads((phase/'evidence-audit-protocol.json').read_text())['ids']
if choice=='full':
 evidence={i:select_context(rows[i],mode='full')[0] for i in ids}
else:
 evidence={p['id']:p['selected_passages'] for p in [json.loads(s) for s in (root/'experiments/apertus-v15-phase2-context-v1'/choice/'validation-evidence.jsonl').read_text().splitlines()]}
notes={
 'ost-v1.1-0019':('juridiquement univoques','French Federal Council/Parliament argument explicitly says current-law formulations are tried, legally unambiguous and established in application. This directly supports the German attributed claim.'),
 'ost-v1.1-0995':('Prorogare la legge COVID è inutile','Italian committee argument says extending COVID law is unnecessary/harmful and preserves powers incompatible with democracy. This contradicts the German attributed committee claim; general Federal Council pro-extension arguments must not be substituted for committee attribution.'),
 'ost-v1.1-0169':('esentare tutte le imprese','Italian summary explicitly says200 francs annually and exemption of all companies. It supports the French claim and distinguishes the initiative from the government300-franc counterproposal.'),
 'ost-v1.1-1159':('750 Millionen Euro','German legal text and overview specify750 million euros, not500. Legal paragraphs on pages20/21 directly contradict the French numeric claim; overview alone is less precise for a claim specifically attributing the voting text.'),
 'ost-v1.1-0333':('drei Viertel','German summary page6 says approximately three quarters imported.75% is numerically equivalent and supports the Italian claim. A correct passage does not guarantee the trained decision head uses the equivalence.'),
 'ost-v1.1-1335':('Kapitaleinkommen auf diverse Arten','German committee argument page17 explicitly says capital income is privileged and proposes fairer taxation. This contradicts the Italian attributed claim; Council and committee positions must remain distinct.'),
 'ost-v1.1-0499':(None,'AHV21 pension-funding claim; development reference is an unrelated2026SSR-fees topic. No supporting or contradicting passage was established in this inspection. An unrelated passage cannot prove full-booklet absence; no gold evidence is manufactured for Neutral.'),
 'ost-v1.1-0664':(None,'Pension reform deferral/2026/social-partner claim; development reference concerns climate law. No supporting or contradicting source passage established. Irrelevance supports a cautious abstention but is not a proof of booklet-wide absence.'),
 'ost-v1.1-0833':(None,'Claim asserts Italian Mediterranean environmental legislation conditional on the Swiss vote. The development reference concerns Swiss individual taxation. No such supporting/contradicting passage established. External-world knowledge is not used to label the statement false.')}
cases=[]
for iid in ids:
 row=rows[iid];needle,note=notes[iid];selected=evidence[iid]
 verified={p['id']:p for p in row['passages']}
 for p in selected:
  original=verified[p['id']]
  assert all(p.get(k)==original.get(k) for k in ('text','page','char_start','char_end','source_sha256'))
 def contains(p):return needle is not None and needle.casefold() in ' '.join(p['text'].split()).casefold()
 located=[p for p in row['passages'] if contains(p)]
 provided=[p for p in selected if contains(p)]
 cases.append({'id':iid,'gold':row['label'],'claim':row['claim'],'claim_language':row['claim_language'],
  'document_language':row['document_language'],'source_pdf_sha256':row['source_pdf_sha256'],
  'selected_source_quote_count':len(selected),'exact_quote_provenance_verified':True,
  'manual_source_interpretation':note,'inspected_direct_source_quotes':located,
  'inspected_direct_quote_present_in_model_input':bool(provided) if needle else None,
  'interpretation':'Presence proves this inspected quote was supplied, not model attribution. Absence does not exclude another valid passage. Neutral has no fabricated supporting proof.'})
direct=[case for case in cases if case['gold']!=1]
result={'inspected_direct_quote_presence':{'n_entailment_contradiction':len(direct),'present_in_input':sum(case['inspected_direct_quote_present_in_model_input'] for case in direct),'caution':'Six predetermined qualitative source inspections, not official evidence scoring or a claim that these are the only valid passages'},'scope':'Nine predetermined, non-blinded development source inspections by assistant; not an independent human evidence evaluation or organizer metric.',
 'context':choice,'cases':cases,'minimal_proof_extraction_implemented':False,
 'neutral_policy':'Return exact candidate model-context passages with explicit relevance/sufficiency caution; no claim that unrelated passages prove whole-document absence.',
 'source_protocol_sha256':hashlib.sha256((phase/'evidence-audit-protocol.json').read_bytes()).hexdigest()}
(phase/'evidence-audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print(json.dumps({'context':choice,'cases':[{k:p[k] for k in ('id','gold','selected_source_quote_count','inspected_direct_quote_present_in_model_input')} for p in cases]},indent=2))
