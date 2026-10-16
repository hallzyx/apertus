import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];phase=root/'experiments/apertus-v15-phase2'
s=json.loads((root/'deployment/v15-selection.json').read_text());f=json.loads((root/s['phase2_results']).read_text());c=json.loads((phase/'context-results.json').read_text());b=json.loads((root/'experiments/budget.json').read_text());m=f['final_validation_metrics'];d=f['bounded_neighbor_diversity_evidence_diagnostics'];cal=m['calibration'];choice=f['final_choice']
lines=['# Phase2: Apertus OST research summary','', '![Verified quality and context comparison](context-tradeoffs.png)','',
'## 1. Document-dependence controls','',
'| Original system input | Validation Macro-F1 |','|---|---:|',
'| Correct booklet |0.894461|','| Claim only, head trained exclusively on train |0.699640|',
'| Booklet from another event, same language; seed42 |0.565341|','| Booklet from another event, same language; seed1337 |0.553328|','| Empty premise, frozen original head |0.210229|','| Generic premise, frozen original head |0.174688|','',
'The drop with wrong booklets supports context dependence. The good result without a document shows class signal in the claims alone; these controls do not demonstrate universal semantic grounding. The original labels measure retention under interventions, not the NLI truth of the new pairs.','',
'Controls for the final head/context: '+', '.join(f"{k}={v['original_label_retention_metrics']['macro_f1']:.6f}" for k,v in f['context_matched_controls'].items())+'. All registered controls required for selection pass.','',
'## 2. Context and observed frontier','',
'| Matched context/head | Macro-F1 | Mean tokens | Cached measured seconds |','|---|---:|---:|---:|']
for name,v in c['comparisons'].items():
 q=v['matched_head'];lines.append(f"|{name}|{q['macro_f1']:.6f}|{q['average_context_tokens']:.0f}|{q['average_latency_seconds']:.3f}|")
full=f['full_recheck']['metrics'];lines.extend([f"|Original full, A6000 recheck|{full['macro_f1']:.6f}|{full['average_context_tokens']:.0f}|{full['average_latency_seconds']:.3f}|",'',
'The observed frontier among static F1/token/latency configurations contains 1k, 2k, 8k and prefix2k. 4k was the registered efficiency choice under additional per-class and per-language constraints; it is not presented as a globally non-dominated point. The 8k alternative was registered separately after a material improvement was observed. Architecture selection uses validation and is not blind discovery. Latencies for compact contexts sum encoding, retrieval and head separately; they exclude start-up, PDF extraction and HTTP.','',
'## 3. Evidence','',
f"Exact text, page, offsets and PDF SHA were verified. The lexical 5-gram overlap with references for {choice} is a development diagnostic, not the official metric. References are never required in production. **{d['evidence_sufficiency_n']}** real inferences were run with only the 1–2 proposed quotes: decision preservation **{d['evidence_decision_preservation_rate']:.1%}**; **{d['evidence_strict_reduction_n']}** contexts were strictly reduced, with preservation **{d['evidence_decision_preservation_when_strictly_reduced']:.1%}**. This measures sensitivity, not human proof of sufficiency or attribution.",'',
'The qualitative inspection of 9 cases per language/class is in evidence-audit.json. Output quotes remain ordered source context, not certified minimal proofs. Neutral requires caution: not finding evidence in the selected context does not prove absence from the whole booklet. The possible scope ambiguity of ost-v1.1-0732 is preserved without changing labels. Neighbor/diversity expansions use 90 fixed cases and are not adopted on the basis of that small sample.','',
'## 4. Frozen architecture','',f'**{choice}**, hybrid E5/BM25 retrieval, whole passages within the native token budget, frozen Apertus v1.5 and a multiclass logistic head trained/calibrated on train only. No LoRA, QLoRA, 70B, quantization or adaptive router.','',
'```mermaid','flowchart LR',' PDF[Booklet PDF] --> Parse[Passages with page and offsets]',' Parse --> Retrieve[E5 and BM25 retrieval]',' Claim[DE FR IT claim] --> Retrieve',f' Retrieve --> Context[{choice} context]',' Context --> Apertus[Frozen Apertus v1.5]',' Apertus --> Head[Head calibrated on train]',' Head --> Label[Class 0 1 2 and confidence]',' Context --> Quotes[Source quotes and pages]',' Label --> Metrics[Tokens and time]','```','',
f"Final validation: F1 **{m['macro_f1']:.6f}**; cross-language **{m['cross_lingual']['macro_f1']:.6f}**; 207 deduplicated pairs **{f['deduplicated_macro_f1']:.6f}**. The CLI on 3 cross-language PDFs and a real frontend upload/prediction request were verified on GPU. The Docker frontend requires the configured native backend; the disposable GPUs were destroyed after the evidence was archived.",'',
'## 5. Calibration','',f"ECE (10 bins) **{cal['ece']:.6f}** · Brier **{cal['brier_score']:.6f}** · NLL **{cal['negative_log_likelihood']:.6f}**. Temperature chosen with train OOF; no recalibration on validation.",'',
'| Confidence | n | Accuracy | Mean confidence |','|---|---:|---:|---:|'])
for q in f['confidence_buckets']:
 fmt=lambda value:f'{value:.3f}' if value is not None else '—'
 lines.append(f"|{q['low']:.2f}–{q['high']:.2f}|{q['n']}|{fmt(q['accuracy'])}|{fmt(q['mean_confidence'])}|")
semantic_path=phase/'semantic-reference-diagnostic.json'
if semantic_path.exists():
 semantic=json.loads(semantic_path.read_text());q=semantic['entailment_contradiction']['conditions'].get(choice)
 if q:lines+=['',f"Complementary semantic diagnostic: presence of the source passage nearest to the E5 reference **{q['top1_presence_rate']:.1%}** on Entailment/Contradiction cases. It is an approximation that depends on the same retriever, not human evidence recall or an official evaluation; see semantic-reference-diagnostic.json."]
adaptive_path=phase/'adaptive-8k-simulation.json'
if adaptive_path.exists():
 adaptive=json.loads(adaptive_path.read_text());q=next(v for v in adaptive['simulations'] if v['threshold']==.95);a=q['metrics']
 lines+=['',f"The 4k→8k simulation with threshold 0.95 reaches F1 **{a['macro_f1']:.6f}**, **{a['average_context_tokens']:.0f}** tokens and **{a['average_latency_seconds']:.3f}s**, escalating **{q['escalation_rate']:.1%}** of the examples. It is a promising alternative, not a negative result. Fixed 8k offers higher F1, higher presence of passages near the reference and a single head/forward; the router lacks its own end-to-end verification and threshold transfer to independent booklets. It is not adopted in production."]
lines+=['', 'The adaptive thresholds were only simulated from cache, counting both forwards when escalating. No additional complexity is adopted. The probabilities of wrong-document controls are not interpreted as calibration of true NLI labels.','',
'## 6. Compute','',f"Total observed Vast spend: **USD{b['actual_spend']:.4f}**. Incremental Phase2 spend: **USD{b['phase2']['additional_observed_spend_usd']:.4f}**. Observed remaining credit: **USD{float(b['account_credit_observed'])+float(b.get('account_balance_observed') or 0):.4f}**. Active instances for this task: **{len(b['active_instance_ids'])}**, destruction verified through the paginated API. The USD2 reserve and the additional USD4.50 cap were kept. Provider accounting is subject to late charges; budget.json and resource-audit.json contain timestamps, prices, duration and per-stage observations.",'',
'## 7. Scientific conclusion','',
f"**The hypothesis receives partial support on this development benchmark.** Frozen Apertus representations allow a lightweight head with F1 {m['macro_f1']:.3f}, and the wrong-booklet controls materially reduce retention. The context drops from roughly 14106 to {m['average_context_tokens']:.0f} tokens without sacrificing validation quality. This justifies the final architecture, not a universal claim of reasoning or generalization.",'',
'Remaining caveats: class signal in the claims, only 3 events and 2 connected validation units, duplicates/paraphrases, possible scope ambiguity and uncertified evidence. Evaluation on new independent booklets is required. The 310 consumed examples were never reopened in Phase2: their historical F1 0.899920 belongs only to the original full system, not to the new compact head. This evidence does not justify spending the remaining credit on LoRA.','',
'Full source: [technical_report.md](../../technical_report.md); metrics: context-results.json, final-results.json and accuracy-results.json when present; errors: selected-error-analysis.jsonl.','',
'The 4k and 8k minimal-evidence audits use different class/language-stratified cohorts; 3/12 versus 9/12 is descriptive, not a paired causal comparison.']
(phase/'research-summary.md').write_text('\n'.join(lines)+'\n')
print('Concise seven-part research summary written')
