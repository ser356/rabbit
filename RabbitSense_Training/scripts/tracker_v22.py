"""Seguimiento LK entre anclas revisables, sin confundir tracking con detección.

No presume fiabilidad en fallos de textura, oclusión o movimiento de cámara.
Los keyframes son referencias humanas aproximadas; se reinicia sobre ellos.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from anchor_boxes import box_for_time

@dataclass
class Track:
    bbox: tuple | None
    source: str
    quality: float
    points: int


def clamp_box(b, w, h):
    if b is None:return None
    x1,y1,x2,y2=map(int,map(round,b))
    x1=max(0,min(w-2,x1));y1=max(0,min(h-2,y1))
    x2=max(x1+2,min(w,x2));y2=max(y1+2,min(h,y2))
    return (x1,y1,x2,y2)


def sample_features(gray,bbox,max_corners=180):
    if bbox is None:return None
    x1,y1,x2,y2=bbox
    mask=np.zeros_like(gray,np.uint8)
    dx=max(2,int((x2-x1)*.14));dy=max(2,int((y2-y1)*.14))
    mask[y1+dy:max(y1+dy+1,y2-dy),x1+dx:max(x1+dx+1,x2-dx)]=255
    return cv2.goodFeaturesToTrack(gray,maxCorners=max_corners,qualityLevel=.008,minDistance=5,mask=mask,blockSize=5)

class AnchoredTracker:
    def __init__(self,anchors):
        self.anchors=anchors
        self.prev=None
        self.points=None
        self.box=None
        self.time=None
        self.fallback_count=0

    def update(self,rgb,time):
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        height,width=gray.shape
        ref,source,gap=box_for_time(self.anchors,time,width,height,max_gap=8.0)
        if source=='manual_reference':
            self.box=ref
            self.points=sample_features(gray,self.box)
            self.prev=gray
            self.time=time
            return Track(ref,'anchor_keyframe',1.,0 if self.points is None else len(self.points))
        candidate=None;quality=0.;n=0
        if self.prev is not None and self.points is not None and len(self.points)>=8 and self.box:
            forward,valid,_=cv2.calcOpticalFlowPyrLK(self.prev,gray,self.points,None,
                winSize=(25,25),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
            if forward is not None and valid is not None:
                keep=valid.ravel().astype(bool)
                old=self.points.reshape(-1,2)[keep]
                new=forward.reshape(-1,2)[keep]
                if len(old)>=8:
                    back,bgood,_=cv2.calcOpticalFlowPyrLK(gray,self.prev,new.reshape(-1,1,2),None,
                        winSize=(25,25),maxLevel=3)
                    if back is not None:
                        err=np.linalg.norm(back.reshape(-1,2)-old,axis=1)
                        ok=(err<2.8)&bgood.ravel().astype(bool)
                        old,new=old[ok],new[ok]
                    if len(old)>=8:
                        shift=new-old
                        median=np.median(shift,axis=0)
                        deviation=np.linalg.norm(shift-median,axis=1)
                        inliers=deviation<max(5.,.02*max(self.box[2]-self.box[0],self.box[3]-self.box[1]))
                        if inliers.sum()>=8:
                            median=np.median(shift[inliers],axis=0)
                            dx,dy=map(float,median)
                            max_step=max(25.,.6*max(self.box[2]-self.box[0],self.box[3]-self.box[1]))
                            if (dx*dx+dy*dy)**.5<max_step:
                                candidate=clamp_box((self.box[0]+dx,self.box[1]+dy,
                                    self.box[2]+dx,self.box[3]+dy),width,height)
                                n=int(inliers.sum())
                                quality=min(1.,n/max(20,len(self.points)))*min(1.,n/20)
        if candidate is not None and ref is not None:
            c=np.array([(candidate[0]+candidate[2])/2,(candidate[1]+candidate[3])/2])
            r=np.array([(ref[0]+ref[2])/2,(ref[1]+ref[3])/2])
            if np.linalg.norm(c-r)>.6*max(ref[2]-ref[0],ref[3]-ref[1]):
                candidate=None
        if candidate is not None and quality>=.14:
            tracked=Track(candidate,'lk_optical_flow',round(quality,3),n)
        elif ref is not None:
            tracked=Track(ref,'anchor_interpolation_fallback',0.,0)
            self.fallback_count+=1
        else:
            tracked=Track(None,'untracked',0.,0)
        self.prev=gray
        self.box=tracked.bbox
        self.points=sample_features(gray,self.box)
        self.time=time
        return tracked


def relative_motion(prev_rgb,rgb,box,prev_box):
    """Movimiento visual relativo al fondo. NO mide exploración ni emoción.

    Flujo denso sobre 320px de ancho; fuera del conejo se estima la deriva de cámara.
    Las magnitudes son índices visuales sin calibración fisiológica.
    """
    if prev_rgb is None or box is None or prev_box is None:
        return {'activity_index':0.,'camera_shift':0.,'body_flow':0.,'activity_quality':'no_previous'}
    h,w=rgb.shape[:2]; sw=320;sh=max(64,round(h*sw/w))
    old=cv2.cvtColor(cv2.resize(prev_rgb,(sw,sh)),cv2.COLOR_RGB2GRAY)
    new=cv2.cvtColor(cv2.resize(rgb,(sw,sh)),cv2.COLOR_RGB2GRAY)
    flow=cv2.calcOpticalFlowFarneback(old,new,None,.5,2,15,2,5,1.2,0)
    x1,y1,x2,y2=box
    x1=int(x1*sw/w);x2=int(x2*sw/w);y1=int(y1*sh/h);y2=int(y2*sh/h)
    x1=max(0,min(sw-2,x1));x2=max(x1+2,min(sw,x2))
    y1=max(0,min(sh-2,y1));y2=max(y1+2,min(sh,y2))
    mask=np.ones((sh,sw),dtype=bool)
    m=max(6,round(max(x2-x1,y2-y1)*.18))
    mask[max(0,y1-m):min(sh,y2+m),max(0,x1-m):min(sw,x2+m)]=False
    bg=flow[mask]
    bg_shift=np.median(bg,axis=0) if len(bg)>100 else np.zeros(2)
    cy1=y1+(y2-y1)//6;cy2=y2-(y2-y1)//6
    cx1=x1+(x2-x1)//6;cx2=x2-(x2-x1)//6
    patch=flow[cy1:cy2,cx1:cx2]
    if patch.size<10:
        return {'activity_index':0.,'camera_shift':float(np.linalg.norm(bg_shift)),
                'body_flow':0.,'activity_quality':'insufficient_pixels'}
    residual=np.linalg.norm(patch-bg_shift,axis=2)
    scale=max(12,((x2-x1)*(y2-y1))**.5)
    score=float(np.percentile(residual,70)/scale)
    return {'activity_index':round(min(score,2.),5),
            'camera_shift':round(float(np.linalg.norm(bg_shift)),3),
            'body_flow':round(float(np.median(residual)),3),
            'activity_quality':'camera_moving' if np.linalg.norm(bg_shift)>2.5 else 'measured_visual_motion'}
