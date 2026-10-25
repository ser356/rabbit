#!/usr/bin/env python3
"""RabbitSense V3.2: human-keyframed tracking; no automatic rabbit recognition."""
import argparse, base64, json, math, subprocess
from pathlib import Path
import cv2
import numpy as np

HTML='''<!doctype html><html lang="es"><meta charset="utf-8"><title>RabbitSense V3.2</title>
<style>body{font:16px system-ui;background:#14171c;color:white;max-width:820px;margin:20px auto}canvas{max-width:100%;height:auto;cursor:crosshair;border:1px solid #888;touch-action:none}button{font-size:16px;padding:9px;margin:6px}input{width:100%}#status{color:#aad7ff}</style>
<h2>RabbitSense V3.2 · Marcar a Cinna</h2><p>Arrastra un rectángulo alrededor de Cinna en cada imagen donde se vea. Si no aparece, pulsa «No visible». Usa «Atrás» para corregir.</p>
<canvas id="c"></canvas><div id="status"></div><p><button onclick="skip()">No visible</button><button onclick="back()">Atrás</button><button onclick="save()">Descargar etiquetas JSON</button></p>
<script>
const items=__ITEMS__; let i=0, marks=Array(items.length).fill(null), imgs=items.map(x=>{let im=new Image();im.src=x.image;return im});
let canvas=document.getElementById('c'),ctx=canvas.getContext('2d'),start=null,rect=null;
function draw(){let im=imgs[i];if(!im.complete){im.onload=draw;return}canvas.width=im.naturalWidth;canvas.height=im.naturalHeight;ctx.drawImage(im,0,0);let r=rect||marks[i];if(r){ctx.strokeStyle='#00ff77';ctx.lineWidth=4;ctx.strokeRect(...r)}document.getElementById('status').textContent=`Imagen ${i+1}/${items.length} · segundo ${items[i].time.toFixed(2)} · ${marks.filter(x=>x!==undefined).length} registros`}
function pos(e){let b=canvas.getBoundingClientRect();return [(e.clientX-b.left)*canvas.width/b.width,(e.clientY-b.top)*canvas.height/b.height]}
canvas.onpointerdown=e=>{start=pos(e);rect=null;canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!start)return;let p=pos(e);rect=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.abs(p[0]-start[0]),Math.abs(p[1]-start[1])];draw()};
canvas.onpointerup=e=>{if(!start)return;start=null;if(rect&&rect[2]>8&&rect[3]>8){marks[i]=rect;rect=null;next()}else{rect=null;draw()}};
function next(){if(i<items.length-1)i++;draw()}function skip(){marks[i]=null;next()}function back(){if(i>0)i--;draw()}
function save(){let a=document.createElement('a');let data=items.map((x,j)=>({time:x.time,box:marks[j]}));a.href=URL.createObjectURL(new Blob([JSON.stringify({video:__VIDEO__,width:canvas.width,height:canvas.height,marks:data},null,2)],{type:'application/json'}));a.download='cinna_marcas_v32.json';a.click()}
window.onkeydown=e=>{if(e.key==='ArrowLeft')back();if(e.key==='ArrowRight')skip()};draw();
</script></html>'''

def prep(args):
    cap=cv2.VideoCapture(str(args.video));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
    if not fps or not n:raise SystemExit('No se pudo abrir el vídeo')
    duration=n/fps;items=[];t=0
    while t<duration:
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read()
        if ok:
            h,w=frame.shape[:2];width=min(600,w);frame=cv2.resize(frame,(width,round(h*width/w)))
            ok,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,76]);
            if ok:items.append({'time':round(t,3),'image':'data:image/jpeg;base64,'+base64.b64encode(buf).decode()})
        t+=args.every
    cap.release();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    html=HTML.replace('__ITEMS__',json.dumps(items)).replace('__VIDEO__',json.dumps(str(Path(args.video).resolve())))
    (out/'marcar_cinna.html').write_text(html,encoding='utf8')
    print(f'Preparadas {len(items)} imágenes. Abre: {out/"marcar_cinna.html"}')
    print('Marca a Cinna y descarga cinna_marcas_v32.json. Después ejecuta render.')

def render(args):
    data=json.loads(Path(args.marks).read_text());video=Path(args.video or data['video']);cap=cv2.VideoCapture(str(video))
    fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(3));h=int(cap.get(4))
    if not fps or not n:raise SystemExit(f'No se pudo abrir {video}')
    marks=data['marks'];valid=[(float(x['time']),x['box']) for x in marks if x['box'] is not None]
    if not valid:raise SystemExit('No hay ninguna caja marcada')
    valid.sort();scale=w/float(data['width']);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    # Interpolation is only allowed between consecutive visible annotations, and never over gaps larger than max-gap.
    import bisect,csv
    times=[x[0] for x in valid];rows=[];outpath=out/'cinna_seguimiento_v32.mp4'
    vw=cv2.VideoWriter(str(outpath),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not vw.isOpened():raise SystemExit('No se pudo crear MP4')
    for k in range(n):
        ok,frame=cap.read()
        if not ok:break
        t=k/fps;idx=bisect.bisect_left(times,t);box=None;status='UNKNOWN'
        if idx<len(times) and abs(times[idx]-t)<1/fps:box=valid[idx][1];status='manual'
        elif 0<idx<len(times) and times[idx]-times[idx-1]<=args.max_gap:
            a,b=valid[idx-1],valid[idx];q=(t-a[0])/(b[0]-a[0]);box=[(1-q)*x+q*y for x,y in zip(a[1],b[1])];status='interpolated'
        # Do not interpolate across explicitly invisible timestamps.
        if box is not None and any(m['box'] is None and min(times[max(0,idx-1)],times[min(len(times)-1,idx)]) < m['time'] < max(times[max(0,idx-1)],times[min(len(times)-1,idx)]) for m in marks):box=None;status='UNKNOWN'
        if box is not None:
            x,y,bw,bh=[round(z*scale) for z in box];x=max(0,min(w-1,x));y=max(0,min(h-1,y));bw=max(1,min(w-x,bw));bh=max(1,min(h-y,bh))
            cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,230,90),3)
            rows.append([round(t,3),status,x,y,bw,bh]);label=status
        else:rows.append([round(t,3),'unknown','','','','']);label='UNKNOWN'
        cv2.putText(frame,f'{t:.1f}s  {label}',(12,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),3)
        cv2.putText(frame,f'{t:.1f}s  {label}',(12,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),1)
        vw.write(frame)
    vw.release();cap.release()
    with (out/'cinna_trayectoria_v32.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['seconds','status','x','y','width','height']);wr.writerows(rows)
    print('Vídeo:',outpath,'\nCSV:',out/'cinna_trayectoria_v32.csv','\nCobertura:',round(100*sum(r[1]!='unknown' for r in rows)/len(rows),1),'% (cajas manuales/interpoladas, no detección automática)')

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare');a.add_argument('--video',required=True);a.add_argument('--out',default='runs/v32');a.add_argument('--every',type=float,default=2.0)
    b=sub.add_parser('render');b.add_argument('--video');b.add_argument('--marks',required=True);b.add_argument('--out',default='runs/v32');b.add_argument('--max-gap',type=float,default=3.0)
    args=p.parse_args();globals()['prep' if args.cmd=='prepare' else 'render'](args)
