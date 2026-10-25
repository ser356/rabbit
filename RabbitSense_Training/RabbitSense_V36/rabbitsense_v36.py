#!/usr/bin/env python3
"""RabbitSense 3.6: local/global camera stabilization + independent body-motion evidence.
Boxes are manually guided, not automatic detections. Motion is not ground truth.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np,pandas as pd

def box_of(r):
    if str(r.status).lower() in ('unknown','not_visible') or pd.isna(r.x) or pd.isna(r.width):return None
    return np.array([float(r.x),float(r.y),float(r.width),float(r.height)],dtype=float)

def inside_mask(shape,boxes,pad=.22):
    h,w=shape; mask=np.full((h,w),255,np.uint8)
    for b in boxes:
        if b is None:continue
        x,y,bw,bh=b; p=max(12,int(max(bw,bh)*pad))
        cv2.rectangle(mask,(max(0,int(x-p)),max(0,int(y-p))),
                      (min(w-1,int(x+bw+p)),min(h-1,int(y+bh+p))),0,-1)
    return mask

def estimate(prev,cur,b0,b1,local):
    h,w=prev.shape;mask=inside_mask((h,w),(b0,b1))
    if local:
        cx=(b0[0]+b0[2]/2+b1[0]+b1[2]/2)/2
        cy=(b0[1]+b0[3]/2+b1[1]+b1[3]/2)/2
        radius=max(110,2.6*max(b0[2],b0[3],b1[2],b1[3]))
        yy,xx=np.ogrid[:h,:w]; mask[((xx-cx)**2+(yy-cy)**2)>radius**2]=0
    pts=cv2.goodFeaturesToTrack(prev,900,.009,6,mask=mask,blockSize=5)
    if pts is None or len(pts)<16:return None,0,0,0
    nxt,st,_=cv2.calcOpticalFlowPyrLK(prev,cur,pts,None,winSize=(23,23),maxLevel=3)
    if nxt is None:return None,0,0,0
    back,st2,_=cv2.calcOpticalFlowPyrLK(cur,prev,nxt,None,winSize=(23,23),maxLevel=3)
    if back is None:return None,0,0,0
    a=pts.reshape(-1,2);b=nxt.reshape(-1,2)
    ok=(st.ravel()==1)&(st2.ravel()==1)&(np.linalg.norm(a-back.reshape(-1,2),axis=1)<1.8)
    ok &= (b[:,0]>=0)&(b[:,0]<w)&(b[:,1]>=0)&(b[:,1]<h)
    a=a[ok];b=b[ok]
    if len(a)<12:return None,len(a),0,0
    M,inl=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=2500,confidence=.995)
    n=int(inl.sum()) if inl is not None else 0
    quality=n/max(len(a),1)
    if M is None or n<12 or quality<.48:return None,len(a),n,quality
    scale=np.hypot(M[0,0],M[1,0]);rot=np.degrees(np.arctan2(M[1,0],M[0,0]))
    if not .72<scale<1.35 or abs(rot)>33:return None,len(a),n,quality
    return M,len(a),n,quality

def body_motion(g0,g1,b0,b1,M):
    """Optical flow residual inside the rabbit, after camera warp; NOT leg detection."""
    h,w=g0.shape
    aligned=cv2.warpAffine(g0,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    x,y,bw,bh=b1
    x0=max(0,int(x+.16*bw));x1=min(w,int(x+.84*bw))
    y0=max(0,int(y+.16*bh));y1=min(h,int(y+.84*bh))
    if x1-x0<24 or y1-y0<24:return None
    a=aligned[y0:y1,x0:x1];b=g1[y0:y1,x0:x1]
    if np.std(a)<9 or np.std(b)<9:return None
    flow=cv2.calcOpticalFlowFarneback(a,b,None,.5,3,17,3,5,1.2,0)
    mag=np.linalg.norm(flow,axis=2)
    return float(np.percentile(mag,65)/max(20,(b0[2]+b1[2])/2))

def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tracks=pd.read_csv(args.tracks).sort_values('seconds').reset_index(drop=True)
    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened():raise RuntimeError('No se puede abrir el vídeo')
    fps=cap.get(cv2.CAP_PROP_FPS); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0:raise RuntimeError('FPS inválidos')
    w,h=int(cap.get(3)),int(cap.get(4));ts=tracks.seconds.to_numpy();samples=[]
    step=max(1,round(fps*args.step))
    for fi in range(0,count,step):
        cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,im=cap.read()
        if not ok:continue
        t=fi/fps;r=tracks.iloc[int(np.argmin(abs(ts-t)))];box=box_of(r)
        samples.append((t,im,box,str(r.status)))
    cap.release();rows=[]
    for i,(t,im,b,source) in enumerate(samples):
        valid=b is not None and b[2]>=12 and b[3]>=12 and b[0]<w and b[1]<h and b[0]+b[2]>0 and b[1]+b[3]>0
        row=dict(seconds=round(t,3),tracking='BOX_OK' if valid else 'BOX_MISSING',track_source=source,
                 movement='unknown',speed_bodywidths_s=None,body_flow_bodywidths_s=None,
                 camera_method='none',background_points=0,background_inliers=0,background_quality=0.,reason='')
        if i==0:row['reason']='first_sample'
        elif not valid:row['reason']='rabbit_not_marked'
        else:
            pt,prev,pb,_=samples[i-1]
            if pb is None:row['reason']='previous_box_missing'
            elif max(b[2]/max(1,pb[2]),pb[2]/max(1,b[2]))>args.max_scale_change:row['reason']='box_scale_jump'
            else:
                g0=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY);g1=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                # Local reference is preferred, but global fallback recovers texture-poor floor.
                options=[]
                for method in ('local','global'):
                    M,n,k,q=estimate(g0,g1,pb,b,method=='local')
                    if M is not None:options.append((method,M,n,k,q))
                if not options:row['reason']='camera_reference_unreliable'
                else:
                    method,M,n,k,q=options[0]
                    row.update(camera_method=method,background_points=n,background_inliers=k,background_quality=round(q,3))
                    old=np.array([pb[0]+pb[2]/2,pb[1]+pb[3]/2,1.]);new=np.array([b[0]+b[2]/2,b[1]+b[3]/2])
                    delta=float(np.linalg.norm(new-M@old)/max(20,(pb[2]+b[2])/2))
                    bm=body_motion(g0,g1,pb,b,M)
                    row['body_flow_bodywidths_s']=round(bm/(t-pt),4) if bm is not None else None
                    if delta>args.max_jump:row['reason']='box_or_parallax_jump'
                    else:
                        speed=delta/(t-pt);row['speed_bodywidths_s']=round(speed,4)
                        row['movement']='locomotion_candidate' if speed>=args.threshold else 'low_motion_candidate'
                        row['reason']='camera_compensated_'+method
        rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(out/'motion_v36.csv',index=False)
    summary={'samples':len(df),'tracking':df.tracking.value_counts().to_dict(),
             'movement':df.movement.value_counts().to_dict(),'reasons':df.reason.value_counts().to_dict(),
             'camera_methods':df.camera_method.value_counts().to_dict(),
             'notes':['body_flow is supporting evidence, not a locomotion label','Interpolated boxes are not independent detections','No validated stress or welfare inference']}
    (out/'summary_v36.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    if args.video_out:
        writer=cv2.VideoWriter(str(out/'motion_v36.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),min(10,fps),(w,h))
        if not writer.isOpened():raise RuntimeError('No se puede crear el MP4')
        for (t,im,b,source),r in zip(samples,rows):
            if b is not None:
                x,y,bw,bh=map(int,b);cv2.rectangle(im,(x,y),(x+bw,y+bh),(70,210,80),3)
            cv2.rectangle(im,(0,0),(w,124),(15,15,15),-1)
            lines=[f'{t:.1f}s TRACKING: {r["tracking"]} ({source})',f'MOVEMENT: {r["movement"]}',
                   f'CAMERA: {r["camera_method"]} ({r["background_inliers"]} inliers)',
                   f'SPEED: {r["speed_bodywidths_s"]}  BODY FLOW: {r["body_flow_bodywidths_s"]}']
            for j,line in enumerate(lines):cv2.putText(im,line,(12,27+j*29),cv2.FONT_HERSHEY_SIMPLEX,.49,(240,240,240),1,cv2.LINE_AA)
            writer.write(im)
        writer.release()
    print('TRACKING:',summary['tracking']);print('MOVEMENT:',summary['movement']);print('CAMERA:',summary['camera_methods']);print('MOTIVOS:',summary['reasons'])
    print('CSV:',out/'motion_v36.csv');print('RESUMEN:',out/'summary_v36.json')
    if args.video_out:print('VIDEO:',out/'motion_v36.mp4')

def main():
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--tracks',required=True)
    p.add_argument('--out',required=True);p.add_argument('--step',type=float,default=.5)
    p.add_argument('--threshold',type=float,default=.2);p.add_argument('--max-jump',type=float,default=1.5)
    p.add_argument('--max-scale-change',type=float,default=1.8);p.add_argument('--video-out',action='store_true')
    run(p.parse_args())
if __name__=='__main__':main()
