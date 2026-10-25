#!/usr/bin/env python3
"""RabbitSense V3.1: open-vocabulary rabbit detection (YOLO-World), reacquisition and visual QA.
No bounding boxes inherited from V2.2. Requires downloading pretrained weights on first run.
"""
import argparse, csv, json, math, sys
from pathlib import Path
import cv2


def main():
    p=argparse.ArgumentParser(description='Deteccion visual real de conejo (YOLO-World)')
    p.add_argument('--video',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--model',default='yolov8s-worldv2.pt')
    p.add_argument('--conf',type=float,default=.12)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--every',type=int,default=1,help='Detectar cada N fotogramas (1 recomendado)')
    p.add_argument('--device',default='auto',help='auto, cpu, mps')
    p.add_argument('--max-frames',type=int,default=0)
    a=p.parse_args()
    try:
        from ultralytics import YOLOWorld
        import torch
    except ImportError:
        sys.exit('Falta ultralytics: .venv/bin/python -m pip install "ultralytics>=8.3,<9"')
    video=Path(a.video).expanduser()
    if not video.exists():sys.exit(f'No existe: {video}')
    out=Path(a.out).expanduser();out.mkdir(parents=True,exist_ok=True)
    device=('mps' if torch.backends.mps.is_available() else 'cpu') if a.device=='auto' else a.device
    print('Cargando YOLO-World; la primera vez puede descargar pesos...',flush=True)
    model=YOLOWorld(a.model)
    model.set_classes(['rabbit'])
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():sys.exit('No se pudo abrir el vídeo')
    fps=cap.get(cv2.CAP_PROP_FPS) or 30
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Video encoding supports even dimensions only.
    w=width+(width%2);h=height+(height%2)
    writer=cv2.VideoWriter(str(out/'rabbit_detected_v31.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():sys.exit('No se pudo crear el vídeo de salida')
    rows=[];last_box=None;last_seen=-999;frame_idx=0;detections=0
    while True:
        ok,frame=cap.read()
        if not ok or (a.max_frames and frame_idx>=a.max_frames):break
        sec=frame_idx/fps
        candidates=[]
        if frame_idx%a.every==0:
            result=model.predict(frame,conf=a.conf,imgsz=a.imgsz,device=device,verbose=False)[0]
            if result.boxes is not None:
                for b in result.boxes:
                    xy=b.xyxy[0].cpu().numpy().tolist();conf=float(b.conf[0].item())
                    x1,y1,x2,y2=map(float,xy)
                    area=(x2-x1)*(y2-y1)
                    if area>width*height*.75 or area<width*height*.0002:continue
                    candidates.append((conf,(x1,y1,x2,y2)))
        chosen=None
        if candidates:
            def score(c):
                conf,box=c
                if last_box is None or frame_idx-last_seen>int(fps*2):return conf
                cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
                lx=(last_box[0]+last_box[2])/2;ly=(last_box[1]+last_box[3])/2
                distance=math.hypot((cx-lx)/width,(cy-ly)/height)
                return conf-.25*distance
            chosen=max(candidates,key=score)
            conf,box=chosen;last_box=box;last_seen=frame_idx;detections+=1
            x1,y1,x2,y2=map(int,box)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,0),3)
            cv2.putText(frame,f'RABBIT {conf:.2f}',(max(0,x1),max(24,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,220,0),2)
            status='detected'
        else:
            conf=None;box=None;status='unknown'
            cv2.putText(frame,'RABBIT NOT DETECTED - UNKNOWN',(20,40),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,255),2)
        cv2.putText(frame,f'{sec:.1f}s',(20,height-20),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        if w!=width or h!=height:frame=cv2.copyMakeBorder(frame,0,h-height,0,w-width,cv2.BORDER_CONSTANT,value=(0,0,0))
        writer.write(frame)
        rows.append({'frame':frame_idx,'time_s':round(sec,4),'status':status,'confidence':round(conf,5) if conf is not None else '', 'x1':round(box[0],1) if box else '', 'y1':round(box[1],1) if box else '', 'x2':round(box[2],1) if box else '', 'y2':round(box[3],1) if box else ''})
        frame_idx+=1
        if frame_idx%100==0:print(f'{frame_idx} fotogramas; detectado {detections}',flush=True)
    cap.release();writer.release()
    with (out/'rabbit_tracks_v31.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['frame','time_s','status','confidence','x1','y1','x2','y2']);wr.writeheader();wr.writerows(rows)
    summary={'input':str(video),'frames':frame_idx,'fps':fps,'detections':detections,'detection_rate':round(detections/frame_idx,4) if frame_idx else 0,'device':device,'model':a.model,'caveats':['Modelo zero-shot: puede confundir objetos con conejos','Sin deteccion se marca unknown: no se inventan cajas','Las detecciones no demuestran locomocion ni conducta','No evalua dolor ni estres']}
    (out/'summary_v31.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False));print('VIDEO:',out/'rabbit_detected_v31.mp4');print('CSV:',out/'rabbit_tracks_v31.csv')
if __name__=='__main__':main()
