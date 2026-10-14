"""Render a source-backed branded intro draft. Requires Pillow and FFmpeg."""
import argparse,json,subprocess,textwrap,hashlib
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
W,H=1920,1080
PAPER='#f3f1e9';INK='#192d25';GREEN='#205742';RED='#b4382d';MUTED='#657269';PALE='#e5eae0'
FONTS=Path('/usr/share/fonts/truetype/dejavu')
def font(n,bold=False):return ImageFont.truetype(str(FONTS/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')),n)
def run(cmd):subprocess.run(cmd,check=True)
def text(d,s,xy,size=30,color=INK,bold=False,width=None):
 f=font(size,bold);lines=[]
 for paragraph in s.split('\n'):
  line=''
  for word in paragraph.split():
   test=(line+' '+word).strip()
   if width and d.textlength(test,font=f)>width and line:lines.append(line);line=word
   else:line=test
  lines.append(line)
 y=xy[1]
 for line in lines:d.text((xy[0],y),line,font=f,fill=color);y+=size*1.23
 return y

def card(d,box,title,subtitle='',accent=GREEN):
 d.rounded_rectangle(box,18,fill='#fffdf7',outline='#d6dbd1',width=2);d.rectangle((box[0],box[1]+18,box[0]+6,box[3]-18),fill=accent)
 y=text(d,title,(box[0]+30,box[1]+28),31,accent,True,width=box[2]-box[0]-60)
 if subtitle:text(d,subtitle,(box[0]+30,y+18),23,MUTED,width=box[2]-box[0]-60)

def slide(scene,index,total,root):
 im=Image.new('RGB',(W,H),PAPER);d=ImageDraw.Draw(im)
 # Brand shell and quiet editorial grid.
 d.rectangle((96,55,135,94),fill=RED);d.rectangle((104,63,127,86),outline=PAPER,width=3)
 text(d,'apertus',(154,53),33,INK,True);text(d,'EVIDENCE LAB',(347,64),20,MUTED)
 text(d,'HACK APERTUS / OST TRACK 2A',(1340,62),21,MUTED)
 d.line((96,125,1824,125),fill='#d0d6cc',width=2)
 text(d,scene['kicker'],(100,173),22,RED,True)
 text(d,f'{index+1:02d} / {total:02d}',(1690,172),24,MUTED)
 kind=scene['type'];sid=scene['id']
 if kind=='benchmark':
  text(d,scene['title'],(100,242),64,INK,True,width=1720)
  if 'series' in scene.get('chart',{}) and scene['chart']['series'][0]['value']<1:
   series=scene['chart']['series'];x=670;y=382;bw=655
   text(d,'Macro-F1 · higher is better',(100,340),24,MUTED)
   for i,row in enumerate(series):
    yy=y+i*122;text(d,row['label'],(100,yy+17),28,INK,width=535)
    d.rounded_rectangle((x,yy,x+bw,yy+69),8,fill='#dfe4d8');d.rounded_rectangle((x,yy,x+bw*row['value'],yy+69),8,fill=RED if i==0 else GREEN)
    text(d,f"{row['value']:.3f}",(x+bw+26,yy+14),35,INK,True)
   text(d,'0',(x,755),23,MUTED);text(d,'1.0',(x+bw-36,755),23,MUTED)
   card(d,(1490,387,1824,705),'0.942','Selected hybrid-8k\nCross-language\nMacro-F1')
   text(d,'Same 276-row development set. Later hybrid selection changes context.',(100,820),25,MUTED,width=1720)
  else:
   rows=scene['chart']['series']
   for i,row in enumerate(rows):
    yy=390+i*175;text(d,row['label'],(100,yy),29,INK)
    d.rounded_rectangle((620,yy,1390,yy+85),8,fill='#dfe4d8');d.rounded_rectangle((620,yy,620+770*row['value']/15000,yy+85),8,fill=RED if i==0 else GREEN)
    text(d,f"{row['value']:,.0f}",(1435,yy+16),45,INK,True)
   text(d,'Prompt-token reduction; not a measured speed or cost reduction.',(100,800),28,MUTED)
 elif kind=='closing':
  text(d,'Read the claim.',(100,275),110,INK,True);text(d,'Check the source.',(100,420),110,GREEN,True)
  d.line((105,574,1630,574),fill=RED,width=9)
  text(d,'Apertus Evidence Lab',(108,645),42,INK,True)
  text(d,'Next: the application walkthrough',(108,721),29,MUTED)
 else:
  y=text(d,scene['title'],(100,255),73,INK,True,width=855)
  text(d,scene['body'],(105,max(535,y+40)),32,MUTED,width=825)
  x=1035
  if kind=='opening':
   for i,lang in enumerate(['DE','FR','IT']):
    yy=260+i*170;card(d,(x+i*20,yy,1800+i*4,yy+125),f'{lang} · OFFICIAL VOTING BOOKLET','Claim → document-grounded decision',RED if i==0 else GREEN)
   text(d,'ONE SOURCE. THREE LANGUAGES.',(x,805),22,MUTED,True)
  elif kind=='problem':
   for i,(label,desc) in enumerate([('0 · SUPPORTED','Entailment'),('1 · NOT ESTABLISHED','Neutral'),('2 · CONTRADICTED','Contradiction')]):
    yy=270+i*180;card(d,(x,yy,1820,yy+142),label,desc,RED if i==2 else GREEN)
  elif kind=='architecture' and sid=='03':
   card(d,(x,250,1818,430),'FROZEN APERTUS v1.5','Primary model · unchanged weights')
   d.line((1420,430,1420,490),fill=RED,width=6)
   card(d,(x,490,1818,675),'TRAINED DECISION HEAD','Internal features → logistic regression')
   text(d,'902 TRAINING EXAMPLES · NO LoRA',(x,745),25,GREEN,True,width=780)
  elif kind=='architecture':
   labels=[('PDF + CLAIM','Preserve source pages'),('HYBRID RETRIEVAL','BM25 + multilingual E5'),('8K CONTEXT → APERTUS','Original source passages')]
   for i,(label,desc) in enumerate(labels):
    yy=250+i*190;card(d,(x,yy,1818,yy+145),label,desc)
    if i<2:d.line((1420,yy+145,1420,yy+190),fill=RED,width=5)
  elif kind=='limits':
   card(d,(x,255,1818,440),'902 TRAIN / 276 VALIDATION','Connected voting-event splits')
   card(d,(x,480,1818,645),'DOCUMENT INTERVENTIONS','Empty input / wrong-event document',RED)
   text(d,'Document dependence ≠ perfect evidence',(x,705),27,MUTED,width=760)
   text(d,'Final 8k needs independent evaluation',(x,775),27,MUTED,width=760)
  elif kind=='demo':
   screenshot=Image.open(root/'track_2a/experiments/apertus-live-demo-en-v1/pass-2-03-result-it.png').convert('RGB');screenshot.thumbnail((790,605),Image.Resampling.LANCZOS)
   d.rounded_rectangle((1016,233,1840,860),18,fill='#fffdf7',outline='#cbd2c7',width=3);im.paste(screenshot,(1032,245));text(d,'ACTUAL APP · REAL NATIVE INFERENCE',(x,800),20,RED,True)
 # Captions occupy their own region, below all slide content.
 d.rectangle((0,915,W,H),fill=INK)
 d.rectangle((0,907,int(W*(index+1)/total),915),fill=RED)
 return im

def srtstamp(t):
 ms=round(t*1000);h=ms//3600000;m=ms//60000%60;s=ms//1000%60;return f'{h:02}:{m:02}:{s:02},{ms%1000:03}'
def main():
 p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,default=Path(__file__).parent);args=p.parse_args();root=Path(__file__).resolve().parents[2];out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
 meta=json.loads((args.build_dir/'timeline.json').read_text());scenes=meta['scenes'];segments=[]
 for i,s in enumerate(scenes):
  image=args.build_dir/f"scene-{s['id']}.png";slide(s,i,len(scenes),root).save(image);video=args.build_dir/f"scene-{s['id']}.mp4";duration=s['duration'];n=round(duration*24)
  filt=f"zoompan=z='min(zoom+0.000012,1.012)':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d={n}:s=1920x1080:fps=24,fade=t=in:st=0:d=0.25,fade=t=out:st={max(0,duration-.25)}:d=0.25,format=yuv420p"
  run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(image),'-vf',filt,'-t',str(duration),'-an','-c:v','libx264','-preset','veryfast','-crf','20','-threads','4',str(video)]);segments.append(video);print('RENDERED',s['id'],flush=True)
 concat=args.build_dir/'concat.txt';concat.write_text('\n'.join("file '"+str(x)+"'" for x in segments));silent=args.build_dir/'silent.mp4';run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(concat),'-c','copy',str(silent)])
 subtitle=out/'subtitles.en.srt';subtitle.write_text('\n\n'.join(f"{i+1}\n{srtstamp(c['start'])} --> {srtstamp(c['end'])}\n"+'\n'.join(textwrap.wrap(c['text'],width=76)) for i,c in enumerate(meta['subtitles']))+'\n')
 video=out/'apertus-intro-draft.mp4'
 ass=out/'subtitles.en.ass'
 header='[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,DejaVu Sans,38,&H00FFFFFF,&H00FFFFFF,&H00252D19,&H00252D19,0,0,0,0,100,100,0,0,1,1,0,2,100,100,43,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'
 def ast(t):
  cs=round(t*100);return f'{cs//360000}:{cs//6000%60:02}:{cs//100%60:02}.{cs%100:02}'
 ass.write_text(header+'\n'.join('Dialogue: 0,'+ast(c['start'])+','+ast(c['end'])+',Default,,0,0,0,,'+'\\N'.join(textwrap.wrap(c['text'],width=76)) for c in meta['subtitles'])+'\n')
 run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(silent),'-i',str(args.build_dir/'narration.wav'),'-i',str(subtitle),'-map','0:v','-map','1:a','-map','2:s','-vf',f"ass={ass.resolve()}",'-c:v','libx264','-preset','veryfast','-crf','20','-threads','4','-c:a','aac','-b:a','160k','-c:s','mov_text','-metadata:s:s:0','language=eng','-metadata:s:a:0','language=eng','-movflags','+faststart','-t',str(meta['duration_seconds']),str(video)])
 info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(video)]));assert any(x['codec_type']=='audio' for x in info['streams']);assert any(x['codec_type']=='subtitle' for x in info['streams'])
 proof={k:v for k,v in meta.items() if k!='subtitles'};proof.update(video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),video_bytes=video.stat().st_size,ffprobe=info,status='creative draft, not final submission',subtitle_format='Burned English captions plus selectable English track and SRT',source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip());(out/'render-manifest.json').write_text(json.dumps(proof,indent=2)+'\n');print('VIDEO_COMPLETE',video,flush=True)
if __name__=='__main__':main()
