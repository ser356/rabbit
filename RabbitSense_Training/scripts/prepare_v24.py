#!/usr/bin/env python3
"""Build an auditable manual review queue from V2.3 source recordings.
Never auto-assign locomotion. Cinna.mov remains an independent evaluation file.
"""
import csv,json,subprocess,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/cinna_review_v23'
DEST=ROOT/'runs/review_v24'
FIELDS=['clip_id','source_video','video_id','start_seconds','duration_seconds','clip_path','label','reviewed','notes']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--make-clips',action='store_true',help='Crear MP4 cortos para revisión manual con ffmpeg');args=ap.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    source=SRC/'etiquetas_40_tramos.csv'
    if not source.exists():raise SystemExit('Primero instala RabbitSense V2.3; falta '+str(source))
    with source.open(encoding='utf-8-sig',newline='') as f:items=list(csv.DictReader(f))
    result=[]
    for i,r in enumerate(items):
        name=r['video_original'];video=(SRC/name).resolve()
        if not video.is_file():
            if r['usar_para_entrenamiento'].strip().lower()!='si':continue
            raise FileNotFoundError(video)
        start=float(r['inicio_segundos']);end=float(r['fin_segundos'])
        if end-start<1.5:continue
        # Prior annotation is a suggestion, never automatically authorized for V2.4.
        result.append({'clip_id':f'v24_{i:04d}','source_video':str(video),'video_id':name,
          'start_seconds':f'{start:.2f}','duration_seconds':f'{end-start:.2f}',
          'clip_path':'','label':'unknown','reviewed':'no',
          'notes':f"V2.3 sugería {r['etiqueta']}; {r['comentario']}"})
    if args.make_clips:
        clipdir=DEST/'clips';clipdir.mkdir(exist_ok=True)
        for row in result:
            targetclip=clipdir/(row['clip_id']+'.mp4')
            if not targetclip.exists():
                subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',row['start_seconds'],'-i',row['source_video'],'-t',row['duration_seconds'],'-an','-c:v','libx264','-preset','veryfast','-crf','25',str(targetclip)],check=True)
            row['clip_path']=str(targetclip)
    target=DEST/'etiquetas_para_revisar.csv'
    if target.exists():
        raise SystemExit(f'{target} ya existe; no se sobrescribe tu trabajo. Edítalo o cámbiale el nombre.')
    with target.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(result)
    print(f'Preparados {len(result)} tramos: {target}')
    print('Revisa visualmente los vídeos y marca reviewed=si SOLO en tramos inequívocos.')
    print('Etiqueta: locomotion (desplazamiento sin olfateo evidente), exploring_sniffing (investigación), resting (quietud sostenida), unknown.')
    print('NO usar cinna.mov para entrenamiento: se reserva para comparación.')
if __name__=='__main__':main()
