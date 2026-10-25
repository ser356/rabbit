"""Interpolación transparente de cajas anotadas manualmente. NO es un detector de conejos."""
from __future__ import annotations
import json
from pathlib import Path


def load_anchors(path: Path):
    raw=json.loads(path.read_text(encoding='utf-8'))
    w=int(raw['reference_width']);h=int(raw['reference_height'])
    if w<50 or h<50:raise ValueError('Dimensiones de referencia no válidas')
    points=raw['anchors']
    if len(points)<2:raise ValueError('Se necesitan al menos dos anclas')
    previous=float('-inf')
    for p in points:
        t=float(p['time']);b=p['bbox']
        if t<=previous:raise ValueError('Anclas desordenadas/duplicadas')
        if len(b)!=4 or not (0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):
            raise ValueError(f'Ancla incorrecta a {t}s: {b}')
        previous=t
    return raw


def box_for_time(anchors: dict, t: float, width: int, height: int, *, max_gap=8.0, max_edge=1.5):
    """(box,source,gap). Entre anclas, estima linealmente. Nunca inventa cajas en huecos enormes."""
    points=anchors['anchors']
    if t < points[0]['time']-max_edge or t > points[-1]['time']+max_edge:
        return None,'uncovered',None
    if t <= points[0]['time']:
        l=r=points[0]
    elif t >= points[-1]['time']:
        l=r=points[-1]
    else:
        import bisect
        ix=bisect.bisect_right([p['time'] for p in points],t)
        l,r=points[ix-1],points[ix]
    gap=float(r['time'])-float(l['time'])
    if gap>max_gap:
        return None,'uncovered_large_gap',gap
    fac=0. if gap==0 else (t-float(l['time']))/gap
    xy=[float(v)+(float(w)-float(v))*fac for v,w in zip(l['bbox'],r['bbox'])]
    scale_x=width/int(anchors['reference_width'])
    scale_y=height/int(anchors['reference_height'])
    x1,y1,x2,y2=map(round,(xy[0]*scale_x,xy[1]*scale_y,xy[2]*scale_x,xy[3]*scale_y))
    x1=max(0,min(width-1,x1));y1=max(0,min(height-1,y1))
    x2=max(x1+1,min(width,x2));y2=max(y1+1,min(height,y2))
    source='manual_reference' if abs(float(l['time'])-t)<1e-7 or abs(float(r['time'])-t)<1e-7 else 'interpolated_reference'
    return (x1,y1,x2,y2),source,round(gap,3)
