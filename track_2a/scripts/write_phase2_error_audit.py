import json,hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ost_nli.data import load_dataset
from ost_nli.model import select_context
root=Path(__file__).resolve().parents[1];phase=root/'experiments/apertus-v15-phase2'
selection=json.loads((root/'deployment/v15-selection.json').read_text());f=json.loads((root/selection['phase2_results']).read_text());choice=f['final_choice']
rows={r['id']:r for r in load_dataset(root/'data/private/full-booklets-v2/validation.jsonl')}
if choice!='full':
 evidence={p['id']:p['selected_passages'] for p in map(json.loads,(root/'experiments/apertus-v15-phase2-context-v1'/choice/'validation-evidence.jsonl').read_text().splitlines())}
else:evidence={iid:select_context(row,mode='full')[0] for iid,row in rows.items()}
records=[]
for error in f['final_validation_errors']:
 row=rows[error['id']];passages=evidence[row['id']];lookup={p['id']:p for p in row['passages']}
 for p in passages:assert all(p.get(k)==lookup[p['id']].get(k) for k in ('text','page','char_start','char_end','source_sha256'))
 records.append({**error,'context_method':choice,'source_pdf_sha256':row['source_pdf_sha256'],
 'evidence':passages,'evidence_role':'Exact model-input context, not validated proof or attribution',
 'suspected_failure_mode':'classification-head mapping' if error['base_correct_head_wrong_same_context'] else 'undetermined',
 'notes':'Restricted native base scorer is correct with identical supplied context while head is wrong. This establishes decision disagreement, not its causal origin or a need for weight fine-tuning.' if error['base_correct_head_wrong_same_context'] else 'No unsupported causal category assigned; historical source hypotheses are qualified and preserved separately.'})
(phase/'selected-error-analysis.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records))
print(json.dumps({'context':choice,'errors':len(records),'base_correct_head_wrong':sum(r['base_correct_head_wrong_same_context'] for r in records),'claims':[{k:r[k] for k in ('id','gold','predicted','claim','suspected_failure_mode')} for r in records]},indent=2,ensure_ascii=False))
