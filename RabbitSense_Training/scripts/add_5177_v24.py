#!/usr/bin/env python3
"""Register IMG_5177.mp4 as reviewed candidates in V2.4 without leaking cinna.mov."""
import argparse,csv,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FIELDS=['clip_id','source_video','video_id','start_seconds','duration_seconds','clip_path','label','reviewed','notes']
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path.home()/'Downloads/IMG_5177.mp4');p.add_argument('--confirm',action='store_true',help='Marcar como revisados los tramos de locomoción 30-41.5 s');a=p.parse_args()
 source=a.input.expanduser().resolve()
 if not source.is_file():sys.exit(f'No encuentro {source}')
 dest=ROOT/'data/cinna_review_v24';dest.mkdir(parents=True,exist_ok=True)
 video=dest/'IMG_5177.mp4'
 if source!=video and not video.exists():shutil.copy2(source,video)
 csvpath=ROOT/'runs/review_v24/etiquetas_para_revisar.csv'
 if not csvpath.exists():sys.exit(f'No encuentro {csvpath}; instala y prepara V2.4 antes')
 with csvpath.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 if not set(FIELDS).issubset(rows[0].keys()):sys.exit('Columnas del CSV no compatibles')
 if any(r['clip_id'].startswith('v24_5177_') for r in rows):sys.exit('IMG_5177 ya está registrado; no duplico filas')
 clips=ROOT/'runs/review_v24/clips';clips.mkdir(parents=True,exist_ok=True)
 # Conservador: dejar 0-30s sin etiqueta; revisar pausas entre intervalos 30-41.5s.
 spans=[(30.,33.5),(34.,37.5),(38.,41.3)]
 for i,(start,end) in enumerate(spans):
  clip=clips/f'v24_5177_{i:02d}.mp4'
  subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(start),'-i',str(video),'-t',str(round(end-start,2)),'-an','-c:v','libx264','-preset','veryfast','-crf','24',str(clip)],check=True)
  rows.append(dict(clip_id=f'v24_5177_{i:02d}',source_video=str(video.resolve()),video_id=video.name,start_seconds=str(start),duration_seconds=str(round(end-start,2)),clip_path=str(clip),label='locomotion',reviewed='si' if a.confirm else 'no',notes='Candidato: desplazamiento siguiendo a persona; comprobar pausas y visibilidad antes de aprobar'))
 backup=csvpath.with_suffix('.before_5177.csv');shutil.copy2(csvpath,backup)
 with csvpath.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
 print('Clips:',clips/'v24_5177_00.mp4','...',clips/'v24_5177_02.mp4')
 print('CSV:',csvpath,'\nCopia anterior:',backup)
 print('Etiquetas nuevas:', 'confirmadas por --confirm' if a.confirm else 'PENDIENTES: revisar los 3 clips y cambiar reviewed a si')
 print('\nSIGUIENTE PASO: extraer características para el nuevo vídeo:')
 print(f'  .venv/bin/python scripts/predict_v22.py --input "{video}" --checkpoint runs/behaviour/best.pt --temporal runs/temporal_v23/best.pt --out runs/predictions_v24/5177 --interval 0.5 --no-video')
 print('\nDespués prueba: .venv/bin/python scripts/train_v24.py')
 print('Nota: train_v24.py exige grabaciones suficientes y validación con las 3 clases; si faltan, se detendrá sin entrenar.')
if __name__=='__main__':main()
