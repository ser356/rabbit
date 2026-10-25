#!/usr/bin/env python3
"""Analiza fotos/vídeos con el clasificador de 7 comportamientos.

NO asigna porcentajes de estrés: estas probabilidades son de las clases visuales.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import tempfile
import subprocess
from collections import Counter
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import timm
import torch
from PIL import Image,ImageOps
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[1]
IMAGES={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
VIDEOS={'.mp4','.mov','.mkv','.avi','.m4v','.webm'}


def device_name():
    return 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')


def load_classifier(checkpoint,device):
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=timm.create_model(ckpt['model_name'],pretrained=False,num_classes=len(ckpt['classes']))
    model.load_state_dict(ckpt['model_state'],strict=True)
    model.to(device).eval()
    normalization=ckpt.get('normalization',{'mean':(.485,.456,.406),'std':(.229,.224,.225)})
    pipeline=transforms.Compose([
        transforms.Resize((256,256),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize(normalization['mean'],normalization['std'])
    ])
    return model,pipeline,ckpt['classes']


def detect_rabbit(image,detector):
    if detector is None:return image
    predictions=detector.predict(source=np.array(image),conf=.25,verbose=False,device=device_name(),imgsz=640)
    if not predictions or predictions[0].boxes is None or len(predictions[0].boxes)==0:
        return image
    boxes=predictions[0].boxes
    i=int(boxes.conf.argmax())
    x1,y1,x2,y2=map(float,boxes.xyxy[i].tolist())
    w,h=image.size
    dx=(x2-x1)*.18;dy=(y2-y1)*.18
    coords=(max(0,int(x1-dx)),max(0,int(y1-dy)),min(w,int(x2+dx)),min(h,int(y2+dy)))
    if coords[2]<=coords[0] or coords[3]<=coords[1]:return image
    return image.crop(coords)


def frames_from_video(video:Path,frame_dir:Path,interval:float):
    exe=imageio_ffmpeg.get_ffmpeg_exe()
    dest=str(frame_dir/'frame_%06d.jpg')
    cmd=[exe,'-hide_banner','-loglevel','error','-y','-i',str(video),
         '-vf',f'fps=1/{interval}', '-q:v','3','-an',dest]
    print('Extrayendo fotogramas con ffmpeg...')
    result=subprocess.run(cmd,capture_output=True,text=True)
    if result.returncode!=0:
        raise RuntimeError('ffmpeg no pudo leer el vídeo: '+result.stderr[-1500:])
    return sorted(frame_dir.glob('frame_*.jpg'))


def predict(args):
    video=args.input.suffix.lower() in VIDEOS
    if not video and args.input.suffix.lower() not in IMAGES:
        raise ValueError('Formato de archivo no admitido. Vídeo: mp4/mov/mkv/avi/m4v/webm. Imagen: jpg/png/webp...')
    if not args.input.is_file():raise FileNotFoundError(args.input)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'{args.checkpoint}. Primero ejecuta start.command para entrenar.')
    device=device_name()
    model,pipeline,labels=load_classifier(args.checkpoint,device)
    print('Modelo:',args.checkpoint,'| GPU/CPU:',device,'| Clases:',labels)
    detector=None
    detector_path=args.detector
    if detector_path is None:
        candidate=ROOT/'runs/detector/rabbit/weights/best.pt'
        if candidate.is_file():detector_path=candidate
    if detector_path is not None:
        if not detector_path.is_file():raise FileNotFoundError(detector_path)
        try:from ultralytics import YOLO
        except ImportError as exc:raise RuntimeError('Instala ultralytics para usar el detector.') from exc
        detector=YOLO(str(detector_path))
        print('Se recortará el conejo usando:',detector_path)
    else:
        print('AVISO: no hay detector; se clasifica TODO el fotograma. '
              'El fondo puede perjudicar mucho el resultado.')
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='rabbitsense_frames_') as temp:
        if video:
            imgs=frames_from_video(args.input,Path(temp),args.interval)
            if len(imgs)>args.max_frames:
                print('AVISO: demasiados frames; solo se analizan los primeros',args.max_frames)
                imgs=imgs[:args.max_frames]
        else: imgs=[args.input]
        if not imgs:raise RuntimeError('No se extrajeron fotogramas del vídeo.')
        with torch.inference_mode():
            for i,path in enumerate(imgs):
                try:
                    with Image.open(path) as im:
                        img=ImageOps.exif_transpose(im).convert('RGB')
                    img=detect_rabbit(img,detector)
                    tensor=pipeline(img).unsqueeze(0).to(device)
                    scores=torch.softmax(model(tensor),dim=-1).squeeze(0).cpu().tolist()
                    idx=max(range(len(scores)),key=lambda j:scores[j])
                    row={'time_seconds':round(i*args.interval,3) if video else 0.0,
                         'behaviour':labels[idx],'confidence':round(scores[idx],5)}
                    row.update({f'prob_{cls}':round(float(v),5) for cls,v in zip(labels,scores)})
                    rows.append(row)
                except (OSError,ValueError) as e:
                    print(f'Error en {path.name}: {e}')
    if not rows:raise RuntimeError('No fue posible analizar ninguna imagen.')
    target=args.out/(args.input.stem+'_behaviours.csv')
    with target.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    counts=Counter(row['behaviour'] for row in rows)
    average={label:round(sum(row[f'prob_{label}'] for row in rows)/len(rows),4) for label in labels}
    summary={'input':str(args.input),'frames':len(rows),'interval_seconds':args.interval if video else 0,
             'detector_used':str(detector_path) if detector is not None else None,
             'behaviour_counts':dict(counts),'mean_class_probabilities':average,
             'caution':'Las probabilidades NO miden estrés ni diagnostican dolor. '
                       'Las clases son reconocimientos visuales entrenados con imágenes, no movimientos temporales.'}
    summary_path=args.out/(args.input.stem+'_summary.json')
    summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print('Comportamientos observados:',dict(counts))
    print('CSV:',target,'\nResumen:',summary_path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'runs/behaviour/best.pt')
    p.add_argument('--detector',type=Path,default=None,help='Modelo YOLOv8 detector de conejo; opcional')
    p.add_argument('--interval',type=float,default=.5,help='Segundos entre fotogramas (default: 0.5)')
    p.add_argument('--max-frames',type=int,default=1000)
    p.add_argument('--out',type=Path,default=ROOT/'runs/predictions')
    args=p.parse_args()
    if args.interval<=0 or args.max_frames<1:p.error('interval y max-frames deben ser positivos')
    predict(args)
if __name__=='__main__':main()
