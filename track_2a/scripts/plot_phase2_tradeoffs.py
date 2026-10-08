"""Plot verified development comparisons; optional tooling, not an app dependency."""
import hashlib,json
from pathlib import Path

def main():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 import numpy as np
 root=Path(__file__).resolve().parents[1];phase=root/'experiments/apertus-v15-phase2'
 selection_path=root/'deployment/v15-selection.json';selection=json.loads(selection_path.read_text());assert selection.get('phase2_completed')
 paths=[phase/'context-results.json',phase/'adaptive-8k-simulation.json',root/selection['phase2_results'],selection_path]
 context,adaptive,final,_=[json.loads(p.read_text()) for p in paths];choice=final['final_choice'];full=final['full_recheck']['metrics']
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
 fig,(left,right)=plt.subplots(1,2,figsize=(12.5,5.1),gridspec_kw={'width_ratios':[1.55,1]})
 offsets={'hybrid-1k':(5,9),'hybrid-2k':(-17,10),'hybrid-4k':(-19,-19),'hybrid-8k':(7,-16),'prefix-2k':(5,7)}
 for name,entry in context['comparisons'].items():
  m=entry['matched_head'];selected=name==choice
  left.scatter(m['average_context_tokens'],m['macro_f1'],s=140 if selected else 55,marker='*' if selected else 'o',color='#d45b24' if selected else '#28759b',zorder=4)
  left.annotate(name.replace('hybrid-',''),(m['average_context_tokens'],m['macro_f1']),xytext=offsets[name],textcoords='offset points',fontsize=10)
 left.scatter(full['average_context_tokens'],full['macro_f1'],s=60,color='#666666',zorder=4)
 left.annotate('Original capped/full',(full['average_context_tokens'],full['macro_f1']),xytext=(-105,10),textcoords='offset points')
 for s in adaptive['simulations']:
  if s['threshold']==.5:continue
  m=s['metrics'];left.scatter(m['average_context_tokens'],m['macro_f1'],s=55,marker='D',facecolors='none',edgecolors='#43815b',zorder=3)
  if s['threshold']==.95:left.annotate('Adaptive t=0.95\n(simulated)',(m['average_context_tokens'],m['macro_f1']),xytext=(-79,12),textcoords='offset points',fontsize=9,color='#326843')
 left.set(xlabel='Mean total prompt tokens (both calls when escalating)',ylabel='Validation Macro-F1',xlim=(0,15300),ylim=(.74,.983),title='Observed quality/context trade-off')
 left.grid(alpha=.2)
 left.legend(handles=[Line2D([],[],marker='o',linestyle='',color='#28759b',label='Static matched heads'),Line2D([],[],marker='*',linestyle='',color='#d45b24',markersize=12,label='Final static selection'),Line2D([],[],marker='D',linestyle='',markerfacecolor='none',color='#43815b',label='Cached adaptive simulations')],loc='lower right',fontsize=8)
 positions=np.arange(3);width=.25
 for index,(name,m,color) in enumerate([('Original full',full,'#858585'),('4k',context['comparisons']['hybrid-4k']['matched_head'],'#65a3bc'),('8k',context['comparisons']['hybrid-8k']['matched_head'],'#d45b24')]):
  right.bar(positions+(index-1)*width,[m['per_class'][str(k)]['f1'] for k in range(3)],width=width,label=name,color=color)
 right.set(xticks=positions,xticklabels=['Entailment','Neutral','Contradiction'],ylim=(.75,1),ylabel='Class F1',title='All three official classes')
 right.tick_params(axis='x',labelsize=9);right.legend(fontsize=9,loc='lower left');right.grid(axis='y',alpha=.2)
 fig.suptitle('Frozen Apertus v1.5: verified development results',fontsize=16,y=.98)
 fig.text(.5,.91,'276 rows · 207 distinct pairs · 3 voting events · no new held-out evaluation',ha='center',fontsize=10,color='#555555')
 fig.text(.02,.02,'Head/temperature fitted on train only. Adaptive points are cached simulations, not a deployed router. No population confidence intervals.',fontsize=9,color='#555555')
 fig.tight_layout(rect=(0,.06,1,.88))
 for extension in ('png','svg'):fig.savefig(phase/f'context-tradeoffs.{extension}',dpi=160)
 plt.close(fig)
 (phase/'figure-source.json').write_text(json.dumps({'sources':[{'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__,'scope':'Development comparisons only; no claim of independent hidden-test superiority.'},indent=2)+'\n')
 print('Scientific PNG/SVG and source hashes written')
if __name__=='__main__':main()
