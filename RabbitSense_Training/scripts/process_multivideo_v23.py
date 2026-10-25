#!/usr/bin/env python3
"""RabbitSense V2.3: trainable multi-recording preparation. No web access required.

Uses exactly the compressed review videos provided by the user (720p, 12 fps)
and individual approximate manual reference boxes for EACH VIDEO.
Unknown-only recordings are not forcibly labelled or trained.
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys, hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCES=ROOT/'data/cinna_review_v23'
ANCHORS=ROOT/'runs/anchors_v23'
FEATURES=ROOT/'runs/features_v22'
LABELS=ROOT/'runs/review/cinna_v23_labels.csv'


def key(video:Path):
    video=video.resolve()
    return video.stem+'_'+hashlib.sha256(str(video).encode()).hexdigest()[:10]


def main():
    ap=argparse.ArgumentParser(description='Extraer características de 7 vídeos de Cinna y preparar las etiquetas para el entrenamiento temporal.')
    ap.add_argument('--force',action='store_true',help='Reprocesar aunque exista la caché')
    args=ap.parse_args()
    meta=SOURCES/'etiquetas_40_tramos.csv'
    if not meta.is_file():raise SystemExit(f'Falta {meta}; instala el ZIP V2.3 dentro del proyecto original')
    with meta.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    by_video=defaultdict(list)
    for r in rows:
        if r['usar_para_entrenamiento'].strip().lower()!='si' or r['etiqueta']=='unknown':continue
        by_video[r['video_original']].append(r)
    print(f'Grabaciones con etiquetas utilizables: {len(by_video)}; grabaciones unknown-only ignoradas')
    if len(by_video)!=7:raise RuntimeError(f'Esperados 7 vídeos útiles, encontramos {len(by_video)}; revisa CSV de origen.')
    assert sum(len(x) for x in by_video.values())==18
    all_labels=[]
    for index,(name,parts) in enumerate(sorted(by_video.items()),1):
        video=(SOURCES/name).resolve();anchor=ANCHORS/(name+'.json')
        if not video.is_file() or not anchor.is_file():
            raise FileNotFoundError(f'Falta vídeo o anclas específicas: {video} / {anchor}')
        with anchor.open(encoding='utf8') as f: metadata=json.load(f)
        if metadata.get('video')!=name:
            raise ValueError(f'Anclas incorrectas para {name}; archivo de otro vídeo.')
        cache=FEATURES/(key(video)+'.npz')
        cached=False
        if cache.exists() and not args.force:
            try:
                with np.load(cache,allow_pickle=False) as z:
                    cached=str(z['source_video'].item())==str(video) and len(z['features'])>3 and z['features'].ndim==2 and 'tracker_sources' in z.files
            except Exception:cached=False
        if not cached:
            print(f'[{index}/7] Extrayendo ViT + movimiento: {name}',flush=True)
            cmd=[sys.executable,str(ROOT/'scripts/predict_v22.py'),
                 '--input',str(video),'--anchors',str(anchor),
                 '--checkpoint',str(ROOT/'runs/behaviour/best.pt'),
                 '--temporal',str(ROOT/'runs/temporal_v23/not-yet-trained.pt'),
                 '--out',str(ROOT/'runs/predictions_v23'), '--interval','0.5','--no-video']
            subprocess.run(cmd,check=True,cwd=ROOT)
        else:print(f'[{index}/7] Caché correcta, reutilizando {name}')
        if not cache.is_file():raise RuntimeError(f'No se creó la caché: {cache}')
        with np.load(cache,allow_pickle=False) as z:
            t=z['time_seconds'];src=z['tracker_sources'].astype(str)
            cover=(src!='untracked').mean()
            if cover<.85:raise RuntimeError(f'{name}: solo {cover:.0%} fotogramas con recorte. Revisar anclas antes de entrenar.')
            print(f'    Frames: {len(t)} | recorte existente: {cover:.0%} | métodos: {dict(Counter(src))}')
        for r in parts:
            start=float(r['inicio_segundos']);end=float(r['fin_segundos'])
            if end<=start:raise RuntimeError(f'Intervalo inválido {name} {start}-{end}')
            all_labels.append({'clip_id':r['clip_id'],'source_video':str(video),
              'video_id':r['id_video_exportado'],'start_seconds':f'{start:.2f}',
              'duration_seconds':f'{end-start:.2f}','clip_path':'','label':r['etiqueta']})
    LABELS.parent.mkdir(parents=True,exist_ok=True)
    with LABELS.open('w',encoding='utf8',newline='') as f:
        fieldnames=['clip_id','source_video','video_id','start_seconds','duration_seconds','clip_path','label']
        writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader();writer.writerows(all_labels)
    info={'n_videos':len(by_video),'n_clips':len(all_labels),'class_counts':dict(Counter(r['label'] for r in all_labels)),
      'quality_notes':['Cajas manuales aproximadas, no detección fiable', 'Vídeos comprimidos de revisión de 720 px',
      'Clips de una misma grabación correlacionados', 'No es estimación de estrés, dolor ni bienestar']}
    (LABELS.parent/'cinna_v23_dataset.json').write_text(json.dumps(info,indent=2,ensure_ascii=False))
    print('\nPreparación finalizada:',LABELS,'\n',info)

if __name__=='__main__':main()
