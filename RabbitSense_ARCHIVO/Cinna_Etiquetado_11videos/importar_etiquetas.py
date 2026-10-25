#!/usr/bin/env python3
"""Resuelve los nombres de los vídeos originales a rutas reales del Mac y crea CSV V2.2.
No ejecuta YOLO ni reaprovecha anclas de otros vídeos.
"""
import argparse,csv,sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description='Importar etiquetas de Cinna (11 videos) a RabbitSense V2.2')
    ap.add_argument('--videos',type=Path,default=Path.home()/'Downloads/Cinna_Videos')
    ap.add_argument('--out',type=Path,default=Path.cwd()/'runs/review/cinna_multivideo_labels.csv')
    args=ap.parse_args()
    meta=Path(__file__).resolve().parent/'etiquetas_40_tramos.csv'
    with meta.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if not args.videos.is_dir():raise SystemExit(f'ERROR: no existe la carpeta {args.videos}')
    index={}
    for f in args.videos.rglob('*'):
        if f.is_file() and f.suffix.lower() in {'.mov','.mp4','.m4v','.mkv'}:
            index.setdefault(f.name.casefold(),[]).append(f.resolve())
    missing=[];ambiguous=[];out=[]
    for row in rows:
        name=row['video_original'];items=index.get(name.casefold(),[])
        if not items:
            missing.append(name);continue
        if len(items)!=1:
            ambiguous.append(name);continue
        src=items[0];start=float(row['inicio_segundos']);end=float(row['fin_segundos'])
        out.append({'clip_id':row['clip_id'], 'source_video':str(src),'video_id':row['id_video_exportado'],
                    'start_seconds':f'{start:.2f}','duration_seconds':f'{end-start:.2f}',
                    'clip_path':'','label':row['etiqueta']})
    if missing or ambiguous:
        if missing:print('Vídeos NO encontrados:',*sorted(set(missing)),sep='\n  ',file=sys.stderr)
        if ambiguous:print('Vídeos repetidos (mismo nombre):',*sorted(set(ambiguous)),sep='\n  ',file=sys.stderr)
        raise SystemExit('No se ha creado el CSV: resuelve los archivos indicados para evitar rutas incorrectas.')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=['clip_id','source_video','video_id','start_seconds','duration_seconds','clip_path','label'])
        wr.writeheader();wr.writerows(out)
    from collections import Counter
    print('CSV creado:',args.out)
    print('Vídeos:',len(set(x['source_video'] for x in out)),'| tramos:',len(out))
    print('Distribución:',dict(Counter(x['label'] for x in out)))
    print('\nIMPORTANTE: aún faltan las características ViT de cada vídeo para entrenar.')
    print('RabbitSense V2.2 exige unas anclas de posición diferentes por vídeo. NO uses runs/anchors/cinna.json en todos.')
    print('El CSV solo aporta etiquetas observacionales; no crea recortes ni características.')
if __name__=='__main__':main()
