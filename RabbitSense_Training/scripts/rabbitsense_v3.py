#!/usr/bin/env python3
"""RabbitSense V3: camera-compensated motion evidence from existing V2.2 tracking CSV.
Not a rabbit detector and not a validated behaviour classifier.
"""
import argparse,csv,json,math
from pathlib import Path
from collections import Counter
import cv2
import numpy as np


def read_tracks(path):
    with open(path,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    required={'time_seconds','bbox_x1','bbox_y1','bbox_x2','bbox_y2','bbox_source','track_quality'}
    if not rows or not required.issubset(rows[0]):raise SystemExit('CSV de tracking V2.2 no válido: faltan '+str(required-set(rows[0]) if rows else required))
    return rows

def affine_background(old,cur,box1,box2):
    """Estimate background camera transform, excluding union of rabbit boxes."""
    h,w=old.shape
    mask=np.full((h,w),255,np.uint8)
    for box in (box1,box2):
        if box is None:continue
        x1,y1,x2,y2=[int(z) for z in box]
        pad=round(.08*max(x2-x1,y2-y1))
        cv2.rectangle(mask,(max(0,x1-pad),max(0,y1-pad)),(min(w-1,x2+pad),min(h-1,y2+pad)),0,-1)
    pts=cv2.goodFeaturesToTrack(old,maxCorners=600,qualityLevel=.008,minDistance=8,mask=mask)
    if pts is None or len(pts)<18:return None,0,0
    nextpts,st,_=cv2.calcOpticalFlowPyrLK(old,cur,pts,None,winSize=(23,23),maxLevel=3)
    if nextpts is None:return None,0,0
    ok=st.ravel().astype(bool)
    a=pts.reshape(-1,2)[ok];b=nextpts.reshape(-1,2)[ok]
    if len(a)<15:return None,0,0
    M,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=2000)
    if M is None:return None,0,0
    fraction=float(inliers.mean()) if inliers is not None else 0
    if fraction<.4:return None,len(a),fraction
    return M,len(a),fraction

def box_at(row,sx,sy,w,h):
    try:
        x1,y1,x2,y2=[float(row[k]) for k in ('bbox_x1','bbox_y1','bbox_x2','bbox_y2')]
    except (ValueError,TypeError):return None
    x1=int(np.clip(x1*sx,0,w-1));x2=int(np.clip(x2*sx,0,w-1))
    y1=int(np.clip(y1*sy,0,h-1));y2=int(np.clip(y2*sy,0,h-1))
    return (x1,y1,x2,y2) if x2-x1>=8 and y2-y1>=8 else None

def center(b):return np.array([(b[0]+b[2])/2,(b[1]+b[3])/2],dtype=float)

def interior_motion(prev,cur,b,M):
    """Median residual LK motion of tracked features in inner bbox, camera compensated."""
    x1,y1,x2,y2=b
    dx=int((x2-x1)*.14);dy=int((y2-y1)*.14)
    mask=np.zeros_like(prev)
    mask[y1+dy:y2-dy,x1+dx:x2-dx]=255
    pts=cv2.goodFeaturesToTrack(prev,maxCorners=180,qualityLevel=.008,minDistance=5,mask=mask)
    if pts is None or len(pts)<8:return None,0
    new,st,_=cv2.calcOpticalFlowPyrLK(prev,cur,pts,None,winSize=(21,21),maxLevel=3)
    if new is None:return None,0
    a=pts.reshape(-1,2)[st.ravel()==1];z=new.reshape(-1,2)[st.ravel()==1]
    if len(a)<8:return None,len(a)
    expected=a@M[:,:2].T+M[:,2]
    residual=np.linalg.norm(z-expected,axis=1)
    return float(np.median(residual)),len(residual)

def frame_at(cap,t,width):
    cap.set(cv2.CAP_PROP_POS_MSEC,1000*float(t))
    ok,frame=cap.read()
    if not ok:return None,None
    h,w=frame.shape[:2]
    newh=max(2,int(round(h*width/w)))
    return cv2.resize(frame,(width,newh),interpolation=cv2.INTER_AREA),(w,h)

def analyze(args):
    rows=read_tracks(args.tracks)
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened():raise SystemExit('No se puede abrir vídeo: '+str(args.video))
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    result=[];prev=None;prevbox=None;prevtime=None;orig=None
    writer=None
    for i,row in enumerate(rows):
        t=float(row['time_seconds'])
        frame,original=frame_at(cap,t,args.width)
        if frame is None:continue
        if orig is None:orig=original
        h,w=frame.shape[:2]
        sx=w/original[0];sy=h/original[1]
        b=box_at(row,sx,sy,w,h)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        method=row.get('bbox_source','')
        trackq=float(row.get('track_quality') or 0)
        d={'time_seconds':t,'track_source':method,'track_quality':trackq,'camera_inliers':0,'camera_inlier_fraction':0.0,'relative_center_px_s':'','body_residual_px_s':'','movement_evidence':'unavailable','reason':'','label':'unknown'}
        if prev is None or prevbox is None or b is None:
            d['reason']='sin fotograma previo o caja';
        else:
            dt=t-prevtime
            if dt<=0:
                d['reason']='tiempo no creciente'
            else:
                M,n,frac=affine_background(prev,gray,prevbox,b)
                d['camera_inliers']=n;d['camera_inlier_fraction']=round(frac,3)
                if M is None:d['reason']='sin transformación fiable de cámara'
                else:
                    predicted=M[:,:2]@center(prevbox)+M[:,2]
                    residual_center=float(np.linalg.norm(center(b)-predicted)/dt)
                    body,nbody=interior_motion(prev,gray,prevbox,M)
                    body_rate=(body/dt) if body is not None else None
                    d['relative_center_px_s']=round(residual_center,2)
                    if body_rate is not None:d['body_residual_px_s']=round(body_rate,2)
                    # Thresholds in pixels/sec at normalized 640px width: heuristic, not trained.
                    if method=='anchor_interpolation_fallback' or row.get('bbox_source')=='untracked':
                        d['reason']='caja interpolada o no seguida: no decidir locomoción'
                    elif trackq<.22:
                        d['reason']='seguimiento de baja calidad'
                    elif body_rate is None or nbody<8:
                        d['reason']='sin suficientes puntos internos'
                    elif residual_center>=args.moving_center and body_rate>=args.moving_body:
                        d['label']='locomotion_candidate';d['movement_evidence']='high';d['reason']='desplazamiento relativo + flujo interno'
                    elif residual_center<=args.still_center and body_rate<=args.still_body:
                        d['label']='low_motion_candidate';d['movement_evidence']='low';d['reason']='bajo movimiento relativo; no implica reposo'
                    else:
                        d['reason']='evidencia ambigua o movimiento local sin desplazamiento'
        result.append(d)
        if args.video_out:
            if writer is None:
                writer=cv2.VideoWriter(str(out/'motion_v3.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),max(1,args.fps), (w,h))
                if not writer.isOpened():raise SystemExit('No se puede crear vídeo anotado')
            if b:
                col=(0,200,0) if d['label']=='locomotion_candidate' else (0,180,230) if d['label']=='unknown' else (180,180,180)
                cv2.rectangle(frame,(b[0],b[1]),(b[2],b[3]),col,2)
            cv2.putText(frame,f"{t:.1f}s {d['label']}",(12,32),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),3)
            cv2.putText(frame,f"{t:.1f}s {d['label']}",(12,32),cv2.FONT_HERSHEY_SIMPLEX,.72,(0,0,0),1)
            writer.write(frame)
        prev=gray;prevbox=b;prevtime=t
        if (i+1)%50==0:print(f'Procesados {i+1}/{len(rows)}')
    cap.release()
    if writer is not None:writer.release()
    if not result:raise SystemExit('No se procesaron fotogramas')
    with (out/'motion_v3.csv').open('w',encoding='utf8',newline='') as f:
        dw=csv.DictWriter(f,fieldnames=list(result[0]));dw.writeheader();dw.writerows(result)
    summary={'video':str(args.video),'tracks':str(args.tracks),'frames':len(result),'counts':dict(Counter(r['label'] for r in result)),'thresholds_px_per_second_at_width':{'width':args.width,'moving_center':args.moving_center,'moving_body':args.moving_body,'still_center':args.still_center,'still_body':args.still_body},'limitations':['Reutiliza cajas aproximadas V2.2: NO es detector automático de conejos','Las cajas de interpolación se abstienen','Flujo dentro de caja puede contener fondo o mano','No distingue exploración de alimentación ni evalúa dolor/estrés','Los umbrales son heurísticos sin calibración']}
    (out/'summary_v3.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
    print('Resumen:',summary['counts']);print('CSV:',out/'motion_v3.csv');print('Informe:',out/'summary_v3.json')
    if args.video_out:print('Vídeo:',out/'motion_v3.mp4')

def main():
    p=argparse.ArgumentParser(description='Análisis de movimiento relativo a cámara usando tracking V2.2 existente')
    p.add_argument('--video',type=Path,required=True);p.add_argument('--tracks',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--width',type=int,default=640)
    p.add_argument('--moving-center',type=float,default=12);p.add_argument('--moving-body',type=float,default=5)
    p.add_argument('--still-center',type=float,default=3);p.add_argument('--still-body',type=float,default=3)
    p.add_argument('--video-out',action='store_true');p.add_argument('--fps',type=float,default=2)
    args=p.parse_args()
    if args.width<160:raise SystemExit('--width debe ser >=160')
    analyze(args)
if __name__=='__main__':main()
