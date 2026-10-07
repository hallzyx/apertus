"""Exact native-head prompt budgets shared by research and document serving."""
from .model import nli_messages, select_context

CONDITIONS={'hybrid-1k':(1024,5),'hybrid-2k':(2048,10),
            'hybrid-4k':(4096,20),'hybrid-8k':(8192,30),'prefix-2k':(2048,None)}


def prompt_tokens(tokenizer, context, claim):
    text=tokenizer.apply_chat_template(nli_messages(context,claim),tokenize=False,
        add_generation_prompt=True,enable_thinking=False)+'{"label":'
    return len(tokenizer(text,add_special_tokens=False)['input_ids'])


def pack_passages(tokenizer,candidates,claim,cap,stop_on_overflow=False):
    if prompt_tokens(tokenizer,'',claim)>cap:
        raise ValueError('Claim exceeds prompt token budget')
    selected=[];context=''
    for passage in candidates:
        trial,text=select_context({'passages':selected+[passage]},mode='full',max_bytes=1000000)
        if prompt_tokens(tokenizer,text,claim)>cap:
            if stop_on_overflow:break
            continue
        selected,context=trial,text
    return selected,context,prompt_tokens(tokenizer,context,claim)


class DocumentPipeline:
    """Booklet+claim only; context policy must match the trained head."""
    def __init__(self,engine,head,retriever=None):
        self.engine=engine;self.head=head;self.retriever=retriever
        self.condition=head['context']
        if self.condition not in ('full',*CONDITIONS):raise ValueError('Unsupported production head context')
        if self.condition.startswith('hybrid-') and retriever is None:
            raise ValueError('Context-matched hybrid head requires pinned retrieval model')

    def predict(self,document,claim):
        import time
        from .data import validate_document
        from .model import label_map
        from .v15 import REPO,REVISION
        if not isinstance(document,dict):raise ValueError('Document must be an object')
        validate_document(document)
        if not isinstance(claim,str) or not claim.strip() or len(claim)>16000:raise ValueError('Invalid claim')
        began=time.perf_counter()
        if self.condition=='full':
            selected,context=select_context(document,mode='full');cap=None
        else:
            cap,depth=CONDITIONS[self.condition]
            candidates=self.retriever.retrieve(document['passages'],claim,k=depth,mode='hybrid') if depth else document['passages']
            selected,context,_=pack_passages(self.engine.tokenizer,candidates,claim,cap,depth is None)
        result=self.engine.infer(nli_messages(context,claim),method='head',head=self.head)
        if result['invalid_output'] or result['label'] not in (0,1,2):raise RuntimeError('Invalid native prediction')
        if cap and result['context_tokens']>cap:raise RuntimeError('Native prompt exceeded context policy')
        label=result['label'];elapsed=(time.perf_counter()-began)*1000
        return {'label':label,'label_name':label_map()[str(label)],'probabilities':result['probabilities'],
            'probability_note':result['probability_note'],'model':REPO,'model_revision':REVISION,
            'decision_method':'head','context_policy':self.condition,'evidence':selected,
            'evidence_ids':[p['id'] for p in selected],'evidence_role':'Exact model input passages; relevance and sufficiency require review',
            'input_tokens':result['context_tokens'],'context_tokens':result['context_tokens'],
            'prompt_token_cap':cap,'model_inference_time_ms':result['latency_seconds']*1000,
            'inference_time_ms':elapsed,'latency_seconds':elapsed/1000,
            'truncated':len(selected)<len(document['passages']) or any(p.get('truncated') for p in selected)}
