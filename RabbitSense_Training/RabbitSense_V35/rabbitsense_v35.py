#!/usr/bin/env python3
"""RabbitSense V3.5: independent track status and camera-relative motion candidates.
Uses V3.2 manually guided boxes; does not identify stress or establish ground truth.
"""
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

def get_box(r):
    if str(r.status).lower() in ('unknown','not_visible') or pd.isna(r.x) or pd.isna(r.width): return None
    return np.array([float(r.x),float(r.y),float(r.width),float(r.height)])

def background_transform(a,b,box_a,box_b):
    """Estimate a camera transform from tracked background corners, excluding the rabbit."""
    h,w=a.shape[:2]
    g0=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY);g1=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    mask=np.full((h,w),255,np.uint8)
    for box in (box_a,box_b):
        if box is None:continue
        x,y,bw,bh=box;pad=max(15,int(.25*max(bw,bh)))
        cv2.rectangle(mask,(max(0,int(x-pad)),max(0,int(y-pad))), (min(w-1,int(x+bw+pad)),min(h-1,int(y+bh+pad))),0,-1)
    corners=cv2.goodFeaturesToTrack(g0,maxCorners=1100,qualityLevel=.012,minDistance=9,mask=mask,blockSize=5)
    if corners is None or len(corners)<25:return None,0,0,'few_background_features'
    nxt,st,_=cv2.calcOpticalFlowPyrLK(g0,g1,corners,None,winSize=(25,25),maxLevel=3,
                                      criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    back,st2,_=cv2.calcOpticalFlowPyrLK(g1,g0,nxt,None,winSize=(25,25),maxLevel=3)
    good=(st.ravel()==1)&(st2.ravel()==1)&(np.linalg.norm(corners.reshape(-1,2)-back.reshape(-1,2),axis=1)<1.5)
    src=corners.reshape(-1,2)[good];dst=nxt.reshape(-1,2)[good]
    valid=(dst[:,0]>=0)&(dst[:,0]<w)&(dst[:,1]>=0)&(dst[:,1]<h)
    src=src[valid];dst=dst[valid]
    if len(src)<20:return None,len(src),0,'few_tracked_background_points'
    # Affine avoids spurious perspective warping when the scene has little texture.
    M,inl=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3.5,
                                      maxIters=2000,confidence=.995)
    n=int(inl.sum()) if inl is not None else 0
    if M is None or n<18 or n/len(src)<.45:return None,len(src),n,'camera_motion_unreliable'
    scale=np.hypot(M[0,0],M[1,0]);rot=np.degrees(np.arctan2(M[1,0],M[0,0]))
    if not .75<scale<1.3 or abs(rot)>30:return None,len(src),n,'camera_transform_extreme'
    return M,len(src),n,'ok'

def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tr=pd.read_csv(args.tracks).sort_values('seconds').reset_index(drop=True)
    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened():raise RuntimeError('No se pudo abrir el vídeo')
    fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(3));h=int(cap.get(4))
    if fps<=0:raise RuntimeError('FPS inválido')
    ts=tr.seconds.to_numpy();samples=[]
    for fi in range(0,count,max(1,round(fps*args.step))):
        cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,im=cap.read()
        if not ok:continue
        t=fi/fps;j=int(np.argmin(np.abs(ts-t)));r=tr.iloc[j]
        samples.append((t,im,get_box(r),str(r.status)))
    cap.release()
    rows=[]
    for i,(t,im,b,st) in enumerate(samples):
        valid=b is not None and b[2]>=12 and b[3]>=12 and b[0]<w and b[1]<h and b[0]+b[2]>0 and b[1]+b[3]>0
        row=dict(seconds=round(t,3),tracking='BOX_OK' if valid else 'BOX_MISSING',track_source=st,
                 movement='unknown',speed_bodywidths_s=None,bg_points=0,bg_inliers=0,reason='')
        if i==0:row['reason']='first_sample'
        elif not valid:row['reason']='rabbit_not_marked'
        else:
            pt,prev,pb,pst=samples[i-1]
            if pb is None:row['reason']='previous_box_missing'
            else:
                ratio=max(b[2]/max(1,pb[2]),pb[2]/max(1,b[2]))
                if ratio>args.max_scale_change:row['reason']='box_scale_jump'
                else:
                    M,n,k,reason=background_transform(prev,im,pb,b)
                    row['bg_points']=n;row['bg_inliers']=k
                    if M is None:row['reason']=reason
                    else:
                        old=np.array([pb[0]+pb[2]/2,pb[1]+pb[3]/2,1])
                        new=np.array([b[0]+b[2]/2,b[1]+b[3]/2])
                        displacement=float(np.linalg.norm(new-M@old))
                        width=max(20.,(pb[2]+b[2])/2)
                        delta=displacement/width
                        row['speed_bodywidths_s']=round(delta/(t-pt),4)
                        if delta>args.max_jump:row['reason']='box_or_parallax_jump'
                        else:
                            row['movement']='locomotion_candidate' if row['speed_bodywidths_s']>=args.threshold else 'low_motion_candidate'
                            row['reason']='camera_compensated'
        rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(out/'motion_v35.csv',index=False)
    summary={'samples':len(df),'tracking':df.tracking.value_counts().to_dict(),
             'movement':df.movement.value_counts().to_dict(),'reasons':df.reason.value_counts().to_dict(),
             'warning':'Movement labels are unverified candidates; boxes are manually guided/interpolated. Not a welfare or stress measure.'}
    (out/'summary_v35.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    if args.video_out:
        writer=cv2.VideoWriter(str(out/'motion_v35.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),min(10.,fps),(w,h))
        if not writer.isOpened():raise RuntimeError('No se pudo crear el MP4')
        for (t,im,b,st),r in zip(samples,rows):
            color=(70,210,80) if r['tracking']=='BOX_OK' else (70,70,230)
            if b is not None:
                x,y,bw,bh=map(int,b);cv2.rectangle(im,(x,y),(x+bw,y+bh),color,3)
            cv2.rectangle(im,(0,0),(w,104),(15,15,15),-1)
            cv2.putText(im,f'{t:.1f}s  TRACKING: {r["tracking"]} ({st})',(14,30),cv2.FONT_HERSHEY_SIMPLEX,.64,color,2)
            mv=r['movement'];mc=(40,190,40) if mv=='locomotion_candidate' else (60,180,230) if mv=='low_motion_candidate' else (90,90,240)
            cv2.putText(im,f'MOVEMENT: {mv}',(14,60),cv2.FONT_HERSHEY_SIMPLEX,.62,mc,2)
            cv2.putText(im,f'camera points: {r["bg_inliers"]}   speed: {r["speed_bodywidths_s"]} bw/s',(14,87),cv2.FONT_HERSHEY_SIMPLEX,.47,(240,240,240),1)
            writer.write(im)
        writer.release()
    print('TRACKING:',summary['tracking']);print('MOVEMENT:',summary['movement']);print('MOTIVOS:',summary['reasons'])
    print('CSV:',out/'motion_v35.csv');print('RESUMEN:',out/'summary_v35.json')
    if args.video_out:print('VIDEO:',out/'motion_v35.mp4')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video',required=True);p.add_argument('--tracks',required=True);p.add_argument('--out',required=True)
    p.add_argument('--step',type=float,default=.5);p.add_argument('--threshold',type=float,default=.2)
    p.add_argument('--max-jump',type=float,default=1.5);p.add_argument('--max-scale-change',type=float,default=1.8)
    p.add_argument('--video-out',action='store_true');run(p.parse_args())
if __name__=='__main__':main()
