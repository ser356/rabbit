#!/usr/bin/env python3
"""YOLOv8n, detector opcional para localizar conejo y poder recortar vídeos."""
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
KEY=os.environ.get('ROBOFLOW_API_KEY','').strip()
if not KEY:
    sys.exit('Falta ROBOFLOW_API_KEY. Este paso es opcional; '
             'la clasificación de comportamientos funciona sin Roboflow.')

from roboflow import Roboflow
from ultralytics import YOLO
import torch

out=ROOT/'data/raw/dataset_b'
out.mkdir(parents=True,exist_ok=True)
yaml=out/'data.yaml'
if not yaml.exists():
    print('Descargando Roboflow RabbitDetect, versión pública 1...')
    rf=Roboflow(api_key=KEY)
    project=rf.workspace('adedeji-oluwadamisola').project('rabbitdetect')
    dataset=project.version(1).download('yolov8',location=str(out))
    yaml=Path(dataset.location)/'data.yaml'
if not yaml.is_file():
    raise RuntimeError('Roboflow no ha descargado el data.yaml. Comprueba los permisos de exportación.')

# Algunas exportaciones Roboflow usan rutas absolutas del sistema de origen.
import yaml as pyyaml
config=pyyaml.safe_load(yaml.read_text())
config['path']=str(yaml.parent.resolve())
# La exportación típica de Roboflow utiliza train/images, valid/images y test/images.
for name in ('train','val','test'):
    if name in config:
        field=str(config[name])
        if field.startswith('../'):
            field=field.replace('../','',1)
        if field.startswith('/') and not Path(field).exists():
            field=field.lstrip('/')
        config[name]=field
fixed=yaml.parent/'data_rabbitsense.yaml'
fixed.write_text(pyyaml.safe_dump(config,sort_keys=False))

device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print('Entrenando detector en:',device)
YOLO('yolov8n.pt').train(data=str(fixed),epochs=int(os.getenv('DETECTOR_EPOCHS','60')),
                          imgsz=640,batch=8,workers=0,device=device,
                          project=str(ROOT/'runs/detector'),name='rabbit',exist_ok=True)
print('Pesos del detector:',ROOT/'runs/detector/rabbit/weights/best.pt')
