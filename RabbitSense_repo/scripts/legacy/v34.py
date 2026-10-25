#!/usr/bin/env python3
"""RabbitSense V3.4: camera-compensated displacement from user-verified V3.2 boxes.
Not a trained behavior model. No emotion/pain inference.
"""
import argparse, csv, json, math
from pathlib import Path
import cv2, numpy as np, pandas as pd

def get_args():
 p=argparse.ArgumentParser()
 p.add_argument('--video',required=True);p.add_argument('--tracks',required=True);p.add_argument('--out',required=True)
 p.add_argument('--step',type=float,default=0.5,help='Seconds between evaluated frames')
 p.add_argument('--threshold',type=float,default=0.20,help='Rabbit body widths per second for locomotion candidate')
 p.add_argument('--video-out',action='store_true');p.add_argument('--max-box-jump',type=float,default=1.5,help='Maximum compensated center jump in body widths per sample');p.add_argument('--max-scale-change',type=float,default=1.8,help='Maximum ratio of consecutive box widths');return p.parse_args()

def box(r):
 if str(r.status)=='unknown' or pd.isna(r.x) or pd.isna(r.width):return None
 return np.array([r.x,r.y,r.width,r.height],dtype=float)

def estimate_bg(prev,cur,b0,b1):
 """Affine prev->cur from background ORB matches, masking rabbit at both ends."""
 h,w=prev.shape[:2];g0=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY);g1=cv2.cvtColor(cur,cv2.COLOR_BGR2GRAY)
 m0=np.full((h,w),255,np.uint8);m1=m0.copy()
 for mask,b in [(m0,b0),(m1,b1)]:
  if b is not None:
   x,y,bw,bh=b;pad=max(12,int(.15*max(bw,bh)))
   cv2.rectangle(mask,(max(0,int(x)-pad),max(0,int(y)-pad)),(min(w,int(x+bw)+pad),min(h,int(y+bh)+pad)),0,-1)
 orb=cv2.ORB_create(nfeatures=1600,fastThreshold=10)
 k0,d0=orb.detectAndCompute(g0,m0);k1,d1=orb.detectAndCompute(g1,m1)
 if d0 is None or d1 is None or len(k0)<15 or len(k1)<15:return None,0,0
 matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d0,d1,k=2)
 good=[a for a,b in matches if a.distance<.72*b.distance]
 if len(good)<12:return None,len(good),0
 src=np.float32([k0[m.queryIdx].pt for m in good]);dst=np.float32([k1[m.trainIdx].pt for m in good])
 M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=3000,confidence=.99)
 inliers=int(mask.sum()) if mask is not None else 0
 if M is None or inliers<15 or inliers/max(1,len(good))<.5:return None,len(good),inliers
 scale=float(np.hypot(M[0,0],M[1,0]));rotation=float(np.degrees(np.arctan2(M[1,0],M[0,0])))
 if not .8<=scale<=1.25 or abs(rotation)>22:return None,len(good),inliers
 return M,len(good),inliers

def main():
 a=get_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 tracks=pd.read_csv(a.tracks).sort_values('seconds').reset_index(drop=True)
 cap=cv2.VideoCapture(a.video)
 if not cap.isOpened():raise SystemExit('No se puede abrir el vídeo')
 fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(3));h=int(cap.get(4))
 step=max(1,round(fps*a.step));ids=list(range(0,frames,step));rows=[];samples=[]
 for i in ids:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,img=cap.read()
  if not ok:continue
  t=i/fps;j=int(np.argmin(np.abs(tracks.seconds.to_numpy()-t)));r=tracks.iloc[j]
  samples.append((t,img,box(r),str(r.status)))
 cap.release()
 for idx,(t,img,b,status) in enumerate(samples):
  row={'seconds':round(t,3),'track_status':status,'label':'unknown','speed_bodywidths_s':None,'bg_matches':0,'bg_inliers':0,'reason':''}
  if idx==0:row['reason']='first_sample'
  else:
   pt,prev,pb,ps=samples[idx-1];dt=t-pt
   if b is None or pb is None:row['reason']='rabbit_not_marked'
   elif max(b[2]/max(pb[2],1),pb[2]/max(b[2],1))>a.max_scale_change:row['reason']='box_scale_discontinuity'
   else:
    M,n,k=estimate_bg(prev,img,pb,b);row['bg_matches']=n;row['bg_inliers']=k
    if M is None:row['reason']='camera_motion_unreliable'
    else:
     old=np.array([pb[0]+pb[2]/2,pb[1]+pb[3]/2,1.]);new=np.array([b[0]+b[2]/2,b[1]+b[3]/2]);expected=M@old
     distance=float(np.linalg.norm(new-expected));body=max(25.,(pb[2]+b[2])/2)
     speed=distance/body/dt;row['speed_bodywidths_s']=round(speed,4)
     if distance/body>a.max_box_jump:row['reason']='box_jump_or_camera_parallax'
     else:
      row['label']='locomotion_candidate' if speed>=a.threshold else 'low_motion_candidate'
      row['reason']='measured_camera_compensated'
  rows.append(row)
 df=pd.DataFrame(rows);df.to_csv(out/'motion_v34.csv',index=False)
 counts=df.label.value_counts().to_dict()
 summary={'video':str(a.video),'sampled_frames':len(df),'fps':fps,'step_seconds':a.step,'threshold_bodywidths_per_second':a.threshold,'counts':counts,'caveats':['Movement candidates, not verified behaviors','Interpolated boxes are not independently detected','Rejects abrupt box jumps, box scaling and unreliable camera transforms','Motion estimate fails when insufficient background matches','Camera perspective and depth changes may create false motion','No stress or pain inference']}
 (out/'summary_v34.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
 if a.video_out:
  writer=cv2.VideoWriter(str(out/'motion_v34.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),min(10.,fps), (w,h))
  if not writer.isOpened():raise RuntimeError('Cannot open video writer')
  colors={'locomotion_candidate':(40,190,40),'low_motion_candidate':(220,160,30),'unknown':(70,70,230)}
  for (t,img,b,status),row in zip(samples,rows):
   label=row['label'];color=colors[label]
   if b is not None:
    x,y,bw,bh=[int(v) for v in b];cv2.rectangle(img,(x,y),(x+bw,y+bh),color,2)
   cv2.rectangle(img,(0,0),(w,77),(20,20,20),-1)
   cv2.putText(img,f'{t:05.1f}s  {label}',(12,29),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
   speed=row['speed_bodywidths_s'];info=f'camera inliers {row["bg_inliers"]}  speed {speed if speed is not None else "NA"} body/s'
   cv2.putText(img,info,(12,58),cv2.FONT_HERSHEY_SIMPLEX,.46,(240,240,240),1)
   writer.write(img)
  writer.release()
 print('Etiquetas:',counts);print('Motivos:',df.reason.value_counts().to_dict());print('CSV:',out/'motion_v34.csv');print('Resumen:',out/'summary_v34.json')
 if a.video_out:print('Video:',out/'motion_v34.mp4')
if __name__=='__main__':main()
