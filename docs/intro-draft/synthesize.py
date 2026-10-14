"""Generate draft narration locally; no external voice API or credentials."""
import argparse,hashlib,json,re,wave,math
from pathlib import Path
import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro

MODEL_SHA='7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5'
VOICES_SHA='bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d'
def save(path,a,rate):
 with wave.open(str(path),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);w.writeframes((np.clip(a,-1,1)*32767).astype('<i2').tobytes())
def main():
 p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--build-dir',type=Path,required=True);args=p.parse_args();args.build_dir.mkdir(parents=True,exist_ok=True)
 model=args.model_dir/'kokoro-v1.0.onnx';voices=args.model_dir/'voices-v1.0.bin'
 for path,sha in [(model,MODEL_SHA),(voices,VOICES_SHA)]:assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
 options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
 kokoro=Kokoro.from_session(ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider']),str(voices))
 scenes=json.loads((Path(__file__).parent/'scenes.json').read_text());timeline=[];subtitles=[];all_audio=[];clock=0;rate=24000
 for scene in scenes:
  parts=[np.zeros(int(rate*.45),dtype=np.float32)];local=.45
  for sentence in re.split(r'(?<=[.!?])\s+',scene['narration']):
   audio,rate=kokoro.create(sentence,voice='af_heart',speed=.98,lang='en-us');duration=len(audio)/rate;parts.append(audio)
   words=sentence.split();group_size=math.ceil(len(words)/math.ceil(len(words)/23));groups=[words[i:i+group_size] for i in range(0,len(words),group_size)];offset=0
   for group in groups:
    d=duration*len(group)/len(words);subtitles.append({'start':clock+local+offset,'end':clock+local+offset+d,'text':' '.join(group)});offset+=d
   parts.append(np.zeros(int(rate*.18),dtype=np.float32));local+=duration+.18
  parts.append(np.zeros(int(rate*.55),dtype=np.float32));audio=np.concatenate(parts);duration=len(audio)/rate;save(args.build_dir/f"scene-{scene['id']}.wav",audio,rate);all_audio.append(audio)
  timeline.append({**scene,'start':clock,'duration':duration});clock+=duration;print('SYNTHESIZED',scene['id'],round(duration,2),flush=True)
 combined=np.concatenate(all_audio);save(args.build_dir/'narration.wav',combined,rate)
 metadata={'voice':'Kokoro v1.0 af_heart, synthetic American English voice','engine':'kokoro-onnx 0.4.9; CPUExecutionProvider','model_sha256':MODEL_SHA,'voices_sha256':VOICES_SHA,'duration_seconds':clock,'peak_amplitude':float(np.max(np.abs(combined))),'subtitle_timing':'Sentence audio timing; within-sentence cues proportionally timed by word count','scenes':timeline,'subtitles':subtitles}
 (args.build_dir/'timeline.json').write_text(json.dumps(metadata,indent=2)+'\n');print('NARRATION_COMPLETE',round(clock,2),flush=True)
if __name__=='__main__':main()
