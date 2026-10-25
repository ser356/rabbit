#!/usr/bin/env python3
"""Genera clips cortos y una página OFFLINE para etiquetar comportamientos por vídeo.

No autogenera etiquetas: una imagen estática no valida que el animal explore.
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
from pathlib import Path
from html import escape
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
LABELS=['exploring_sniffing','eating','resting','locomotion','digging','grooming',
        'jumping_binkying','periscoping','humping_mating','other','unknown']


def video_duration(video:Path,exe:str)->float:
    p=subprocess.run([exe,'-hide_banner','-i',str(video)],capture_output=True,text=True)
    match=re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)',p.stderr)
    if not match:raise RuntimeError('No se pudo determinar la duración; usa --duration SEGUNDOS')
    return int(match[1])*3600+int(match[2])*60+float(match[3])


def run(args):
    video=args.input.expanduser().resolve()
    if not video.is_file():raise FileNotFoundError(video)
    exe=imageio_ffmpeg.get_ffmpeg_exe()
    duration=args.duration or video_duration(video,exe)
    out=args.out or ROOT/'runs/review'/video.stem
    clips=out/'clips';clips.mkdir(parents=True,exist_ok=True)
    entries=[]
    for i,start in enumerate(range(0,int(duration),args.clip_seconds)):
        length=min(args.clip_seconds,duration-start)
        if length<.75:break
        dest=clips/f'clip_{i:04d}.mp4'
        if not dest.exists() or dest.stat().st_size<500:
            cmd=[exe,'-hide_banner','-loglevel','error','-y','-ss',str(start),
                 '-i',str(video),'-t',str(length),'-vf','scale=640:-2,fps=12',
                 '-an','-c:v','libx264','-preset','veryfast','-crf','29',
                 '-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
            res=subprocess.run(cmd,capture_output=True,text=True)
            if res.returncode!=0:raise RuntimeError('ffmpeg: '+res.stderr[-800:])
        entries.append({'clip_id':f'{i:04d}','start_seconds':float(start),
                        'duration_seconds':round(length,2),
                        'clip_path':str(dest.relative_to(out)),
                        'source_video':str(video),'video_id':video.stem})
        print('Clip',i+1,'/',int(duration)//args.clip_seconds+1,end='\r',flush=True)
    (out/'review_manifest.json').write_text(json.dumps(entries,indent=2,ensure_ascii=False))
    html='''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RabbitSense V2.2 · Etiquetar acciones</title><style>
body{font-family:system-ui,Arial;background:#151719;color:#f4f5f6;margin:0;padding:30px;line-height:1.55}
main{max-width:1120px;margin:auto}a{color:#8fc3ff}.info{background:#2b3035;padding:16px;border-radius:14px}
#cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px;margin-top:24px}
.card{background:#232629;padding:13px;border-radius:14px;border:1px solid #444}
video{width:100%;border-radius:9px;max-height:370px;background:black}
select{width:100%;padding:12px;color:#fff;background:#30373a;border:1px solid #777;border-radius:9px;margin-top:9px}
button{background:#d8eeff;color:#101820;font-weight:700;border:0;padding:13px 22px;border-radius:9px;cursor:pointer}
.muted{color:#bbc2c5}.small{font-size:13px}
</style></head><body><main><h1>RabbitSense · Etiquetado temporal</h1>
<div class="info"><p><b>Observa el movimiento en cada clip antes de asignar una etiqueta.</b> "Exploring/sniffing" = olfateo y exploración visible con movimiento de cabeza/cuerpo; "resting" = reposo real; "locomotion" = desplazamiento sin acción más específica. Etiqueta mirando el vídeo, no la silueta. Marca unknown cuando no estés seguro.</p>
<p>Las etiquetas no miden emociones, estrés ni dolor. Guarda el CSV en la carpeta indicada en el README. Los vídeos no se suben a ninguna web.</p>
<button onclick="exportLabels()">Descargar rabbitsense_labels.csv</button> <span id="status" class="muted"></span></div>
<div id="cards"></div></main><script>
const items=MANIFEST;
const labels=LABELS;
const cards=document.getElementById('cards');
for(const clip of items){
 const card=document.createElement('article');card.className='card';
 const header=document.createElement('p');header.textContent=`Clip ${clip.clip_id} · ${clip.start_seconds}s–${(clip.start_seconds+clip.duration_seconds).toFixed(1)}s`;
 const video=document.createElement('video');video.src=clip.clip_path;video.controls=true;video.preload='metadata';video.muted=true;video.playsInline=true;
 const sel=document.createElement('select');sel.id='label_'+clip.clip_id;
 for(const [key,title] of [['','— Sin etiquetar —'],...labels.map(x=>[x,x.replaceAll('_',' ')])]){
  const option=document.createElement('option');option.value=key;option.textContent=title;sel.append(option);
 }
 card.append(header,video,sel);cards.append(card);
}
function csvCell(value){return '"'+String(value??'').replaceAll('"','""')+'"';}
function exportLabels(){
 const cols=['clip_id','source_video','video_id','start_seconds','duration_seconds','clip_path','label'];
 const lines=[cols.map(csvCell).join(',')];let valid=0;
 for(const clip of items){
  const label=document.getElementById('label_'+clip.clip_id).value;
  if(!label)continue;valid++;
  lines.push(cols.map(c=>csvCell(c==='label'?label:clip[c])).join(','));
 }
 if(!valid){alert('Primero etiqueta al menos un clip.');return;}
 const blob=new Blob([lines.join('\\r\\n')+'\\r\\n'],{type:'text/csv;charset=utf-8'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='rabbitsense_labels.csv';a.click();
 setTimeout(()=>URL.revokeObjectURL(a.href),1500);
 document.getElementById('status').textContent=`${valid} clips etiquetados, ${items.length-valid} sin etiquetar`;
}
</script></body></html>'''
    html=html.replace('MANIFEST',json.dumps(entries,ensure_ascii=False)).replace('LABELS',json.dumps(LABELS))
    (out/'review.html').write_text(html,encoding='utf8')
    print(f'\n{len(entries)} clips listos, duración {duration:.1f}s')
    print('Abre:',out/'review.html')
    print('En macOS: open',out/'review.html')
    print('Tras etiquetar, guarda rabbitsense_labels.csv en:',out)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=Path.home()/'Downloads/cinna.mov')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--clip-seconds',type=int,default=4)
    parser.add_argument('--duration',type=float,default=None)
    args=parser.parse_args()
    if args.clip_seconds<2 or args.clip_seconds>60:parser.error('--clip-seconds debe estar entre 2 y 60')
    run(args)

if __name__=='__main__':main()
