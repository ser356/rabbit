#!/usr/bin/env python3
"""RabbitSense V2.2 — tracking visual + postura ViT + modelo temporal opcional.

El análisis temporal SOLO predice clases aprendidas con clips etiquetados.
No estima estrés ni dolor. Cajas LK pueden fallar: inspeccionar vídeo anotado.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil,subprocess,tempfile
from collections import Counter
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageOps
from predict import ROOT,frames_from_video,load_classifier,device_name
from anchor_boxes import load_anchors
from tracker_v22 import AnchoredTracker,relative_motion
from temporal_v22 import window_features,load_action_model


def key_for_video(path):
    full=str(Path(path).expanduser().resolve())
    return Path(path).stem+'_'+hashlib.sha256(full.encode()).hexdigest()[:10]


def crop_pil(img,box,margin=.08):
    x1,y1,x2,y2=box
    dx=(x2-x1)*margin;dy=(y2-y1)*margin
    return img.crop((max(0,int(x1-dx)),max(0,int(y1-dy)),
                     min(img.width,int(x2+dx)),min(img.height,int(y2+dy))))


def draw(rgb,row,box):
    h,w=rgb.shape[:2];scale=min(1,960/w)
    dest=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    if scale<1:dest=cv2.resize(dest,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
    if box:
        x1,y1,x2,y2=[round(v*scale) for v in box]
        color=(60,210,80) if row['bbox_source'] in ('anchor_keyframe','lk_optical_flow') else (0,190,255)
        cv2.rectangle(dest,(x1,y1),(x2,y2),color,2)
    texts=[f"{row['time_seconds']:.1f}s | POSTURA (foto): {row['posture_visual']} ({row['posture_confidence']:.2f})",
           f"ACCION (video): {row['action_temporal']} ({row['action_confidence']:.2f})",
           f"Track: {row['bbox_source']} ({row['track_quality']:.2f}); movimiento: {row['activity_index']:.3f}"]
    for i,line in enumerate(texts):
        y=28+i*28
        cv2.rectangle(dest,(0,y-21),(min(dest.shape[1]-1,830),y+7),(22,22,22),-1)
        cv2.putText(dest,line,(9,y),cv2.FONT_HERSHEY_SIMPLEX,.56,(255,255,255),1,cv2.LINE_AA)
    return dest


def evaluate_action_windows(rows,feats,ckpt_path,device,interval,min_conf,min_margin):
    for row in rows:
        row['action_temporal']='untrained' if ckpt_path is None else 'unknown'
        row['action_confidence']=0.
        row['action_margin']=0.
    if ckpt_path is None:return None
    net,ckpt=load_action_model(ckpt_path,device)
    if len(feats[0]) != ckpt['feat_dim']:
        raise RuntimeError('Modelo temporal incompatible con este ViT. Reentrena con train_temporal_v22.command.')
    t=np.array([r['time_seconds'] for r in rows],dtype='float64')
    matrix=np.stack(feats)
    wins=[];valid=[]
    clip_s=float(ckpt.get('clip_seconds',4.))
    for i,row in enumerate(rows):
        end=row['time_seconds']
        start=end-clip_s+interval
        if start<0 or end<=start or end>t[-1]:continue
        try:
            wins.append(window_features(matrix,t,start,end,int(ckpt.get('frames_per_clip',8))))
            valid.append(i)
        except ValueError:continue
    if wins:
        with torch.inference_mode():
            for k in range(0,len(wins),32):
                tensor=torch.tensor(np.stack(wins[k:k+32]),device=device)
                scores=torch.softmax(net(tensor),-1).cpu().numpy()
                for idx,p in zip(valid[k:k+32],scores):
                    top=p.argsort()[-2:][::-1]
                    conf=float(p[top[0]]); margin=float(p[top[0]]-p[top[1]])
                    rows[idx]['action_confidence']=round(conf,5)
                    rows[idx]['action_margin']=round(margin,5)
                    if conf>=min_conf and margin>=min_margin:
                        rows[idx]['action_temporal']=ckpt['classes'][int(top[0])]
    return {'checkpoint':str(ckpt_path),'training_provenance':ckpt.get('training_provenance','unknown'),
            'classes':ckpt['classes']}


def run(args):
    video=args.input.expanduser().resolve()
    if not video.is_file():raise FileNotFoundError(video)
    if not args.checkpoint.is_file():raise FileNotFoundError(args.checkpoint)
    if not args.anchors.is_file():raise FileNotFoundError(f'Faltan anclas para el vídeo: {args.anchors}')
    anchors=load_anchors(args.anchors)
    device=device_name()
    model,transform,classes=load_classifier(args.checkpoint,device)
    print('ViT:',args.checkpoint,'| Dispositivo:',device)
    print('Tracking: LK entre anclas (no YOLO; las anclas son aproximadas).')
    temporal=args.temporal if args.temporal is not None else ROOT/'runs/temporal_v22/best.pt'
    if not temporal.is_file():temporal=None
    print('Acciones temporales:',temporal or 'sin modelo; mostrar untrained')
    out=args.out.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    image_dir=out/(video.stem+'_frames_v22')
    if not args.no_video:image_dir.mkdir(exist_ok=True)
    tracker=AnchoredTracker(anchors)
    rows=[];feats=[];previous=None
    with tempfile.TemporaryDirectory(prefix='rabbitsense_v22_') as scratch:
        frames=frames_from_video(video,Path(scratch),args.interval)
        if not frames:raise RuntimeError('Sin fotogramas')
        if len(frames)>args.max_frames:raise RuntimeError(f'Demasiados fotogramas: {len(frames)} > {args.max_frames}')
        for i,path in enumerate(frames):
            t=round(i*args.interval,3)
            with Image.open(path) as im:img=ImageOps.exif_transpose(im).convert('RGB')
            rgb=np.asarray(img)
            old_box=tracker.box
            tr=tracker.update(rgb,t)
            movement=relative_motion(previous,rgb,tr.bbox,old_box)
            probs=np.zeros(len(classes),dtype='float32')
            if tr.bbox is not None:
                tensor=transform(crop_pil(img,tr.bbox,args.crop_margin)).unsqueeze(0).to(device)
                with torch.inference_mode():
                    z=model.forward_features(tensor)
                    embedding=model.forward_head(z,pre_logits=True)
                    logits=model.forward_head(z)
                    probs=torch.softmax(logits,dim=-1).flatten().cpu().numpy().astype('float32')
                    visual_feature=embedding.flatten().cpu().numpy().astype('float32')
                if visual_feature.ndim!=1:raise RuntimeError('ViT no devuelve embeddings 1D')
                rank=np.argsort(probs)[::-1]
                conf=float(probs[rank[0]]);margin=float(probs[rank[0]]-probs[rank[1]])
                raw=classes[int(rank[0])]
                posture=raw if conf>=args.posture_conf and margin>=args.posture_margin else 'unknown'
            else:
                if not feats:raise RuntimeError('Sin recorte inicial: proporciona un ancla en el segundo 0.')
                visual_feature=np.zeros(len(feats[-1])-4-len(classes),dtype='float32')
                raw='unknown';posture='unknown';conf=margin=0.
            # Cache de características: embeddings ViT, probabilidades fotográficas y movimiento real.
            # Nunca utiliza la traslación de cajas interpoladas como medida observada.
            motion_feats=np.array([min(movement['activity_index'],1.),
                                   min(movement['camera_shift']/10.,1.),tr.quality,
                                   ((tr.bbox[2]-tr.bbox[0])*(tr.bbox[3]-tr.bbox[1])/(img.width*img.height)) if tr.bbox else 0.],dtype='float32')
            feats.append(np.concatenate((visual_feature,probs,motion_feats)))
            row={'time_seconds':t,'posture_raw':raw,'posture_visual':posture,
                 'posture_confidence':round(conf,5),'posture_margin':round(margin,5),
                 'bbox_source':tr.source,'track_quality':round(tr.quality,4),
                 'track_points':tr.points,'activity_index':movement['activity_index'],
                 'camera_shift':movement['camera_shift'],'activity_quality':movement['activity_quality'],
                 'bbox_x1':tr.bbox[0] if tr.bbox else '', 'bbox_y1':tr.bbox[1] if tr.bbox else '',
                 'bbox_x2':tr.bbox[2] if tr.bbox else '', 'bbox_y2':tr.bbox[3] if tr.bbox else ''}
            row.update({f'prob_{cl}':round(float(p),5) for cl,p in zip(classes,probs)})
            rows.append(row)
            previous=rgb
            if (i+1)%25==0 or i==len(frames)-1:print(f'Fotogramas: {i+1}/{len(frames)}')
    provenance=evaluate_action_windows(rows,feats,temporal,device,args.interval,args.action_conf,args.action_margin)
    cache_path=ROOT/'runs/features_v22'/(key_for_video(video)+'.npz')
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache_path,features=np.stack(feats),
                        time_seconds=np.asarray([x['time_seconds'] for x in rows],dtype=np.float32),
                        source_video=str(video),posture_classes=np.asarray(classes),
                        tracker_sources=np.asarray([r['bbox_source'] for r in rows]))
    csvpath=out/(video.stem+'_behaviours_v22.csv')
    with csvpath.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tracking=dict(Counter(r['bbox_source'] for r in rows))
    info={'source_video':str(video),'n_frames':len(rows),'interval_seconds':args.interval,
          'posture_counts':dict(Counter(r['posture_visual'] for r in rows)),
          'action_counts':dict(Counter(r['action_temporal'] for r in rows)),
          'tracking_methods':tracking,'feature_cache':str(cache_path),
          'temporal_model':provenance,'warnings':[
              'ViT entrenado con fotografías: posturas NO son acciones temporales.',
              'Cajas LK + anclas pueden fallar y no constituyen un detector validado.',
              'El movimiento visual no demuestra olfateo ni exploración.',
              'Acciones solo aprendidas si se aporta modelo temporal entrenado con etiquetas reales.',
              'No se mide estrés, dolor ni bienestar emocional.']}
    jpath=out/(video.stem+'_summary_v22.json')
    jpath.write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf8')
    video_path=None
    if not args.no_video:
        # Segunda lectura de fotogramas para evitar almacenar un vídeo entero en memoria.
        # Las cajas y etiquetas ya se calcularon; generar anotaciones sin volver a inferir.
        with tempfile.TemporaryDirectory(prefix='rabbitsense_v22_annotation_') as scratch:
            frames=frames_from_video(video,Path(scratch),args.interval)
            for i,path in enumerate(frames):
                with Image.open(path) as im: rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
                row=rows[i]
                box=tuple(int(row[k]) for k in ('bbox_x1','bbox_y1','bbox_x2','bbox_y2')) if row['bbox_x1']!='' else None
                cv2.imwrite(str(image_dir/f'{i:06d}.jpg'),draw(rgb,row,box),[int(cv2.IMWRITE_JPEG_QUALITY),84])
        import imageio_ffmpeg
        video_path=out/(video.stem+'_annotated_v22.mp4')
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        cmd=[ffmpeg,'-hide_banner','-loglevel','error','-y','-framerate',str(1/args.interval),
             '-i',str(image_dir/'%06d.jpg'),'-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',
             '-c:v','libx264','-preset','veryfast','-crf','25','-pix_fmt','yuv420p',
             '-movflags','+faststart',str(video_path)]
        res=subprocess.run(cmd,capture_output=True,text=True)
        if res.returncode:raise RuntimeError('Error ffmpeg MP4: '+res.stderr[-1400:])
        if not args.keep_frames:shutil.rmtree(image_dir)
    print('Posturas:',info['posture_counts'],'\nAcciones:',info['action_counts'])
    print('Tracking:',tracking)
    print('CSV:',csvpath,'\nResumen:',jpath,'\nFeatures:',cache_path)
    if video_path:print('Vídeo:',video_path)


def main():
    p=argparse.ArgumentParser(description='RabbitSense V2.2: tracking+postura+acciones entrenadas.')
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'runs/behaviour/best.pt')
    p.add_argument('--anchors',type=Path,default=ROOT/'runs/anchors/cinna.json')
    p.add_argument('--temporal',type=Path,default=None)
    p.add_argument('--out',type=Path,default=ROOT/'runs/predictions_v22')
    p.add_argument('--interval',type=float,default=.5)
    p.add_argument('--crop-margin',type=float,default=.08)
    p.add_argument('--posture-conf',type=float,default=.55)
    p.add_argument('--posture-margin',type=float,default=.10)
    p.add_argument('--action-conf',type=float,default=.62)
    p.add_argument('--action-margin',type=float,default=.12)
    p.add_argument('--max-frames',type=int,default=2000)
    p.add_argument('--no-video',action='store_true')
    p.add_argument('--keep-frames',action='store_true')
    a=p.parse_args()
    if a.interval<=0 or a.max_frames<1:p.error('Intervalo y max-frames deben ser positivos')
    run(a)

if __name__=='__main__':main()
