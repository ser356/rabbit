#!/usr/bin/env python3
"""Compare V2.3 and V2.4 on cinna.mov without treating V2.3 outputs as truth."""
import argparse,csv,json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
def load(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser();p.add_argument('--v23',type=Path,default=ROOT/'runs/predictions_v23/cinna/cinna_behaviours_v22.csv');p.add_argument('--v24',type=Path,default=ROOT/'runs/predictions_v24/cinna/cinna_behaviours_v22.csv');p.add_argument('--labels',type=Path,default=ROOT/'runs/review_v24/cinna_test_labels.csv');args=p.parse_args()
    for path in (args.v23,args.v24):
        if not path.exists():raise SystemExit(f'Falta {path}; ejecuta el predictor con ambos checkpoints')
    a,b=load(args.v23),load(args.v24)
    x={round(float(r['time_seconds']),3):r for r in a};y={round(float(r['time_seconds']),3):r for r in b}
    keys=sorted(set(x)&set(y))
    if not keys:raise SystemExit('No hay fotogramas coincidentes entre ejecuciones')
    print('Fotogramas comunes:',len(keys));print('V2.3:',dict(Counter(x[k]['action_temporal'] for k in keys)));print('V2.4:',dict(Counter(y[k]['action_temporal'] for k in keys)))
    print('Cambios de predicción:',sum(x[k]['action_temporal']!=y[k]['action_temporal'] for k in keys))
    if not args.labels.exists():
        print('Sin etiquetas independientes revisadas: NO es posible calcular exactitud.');print('Para ello crea cinna_test_labels.csv con columnas start_seconds,duration_seconds,label,reviewed y reviewed=si.');return
    intervals=load(args.labels);valid=[]
    for r in intervals:
        if r.get('reviewed','').lower() not in ('si','sí','yes','true'):continue
        if r.get('label') not in ('exploring_sniffing','locomotion','resting'):continue
        st=float(r['start_seconds']);en=st+float(r['duration_seconds'])
        valid.append((st,en,r['label']))
    if not valid:print('No hay intervalos test revisados: no se calcula exactitud');return
    for name,pred in [('V2.3',x),('V2.4',y)]:
        truth=[];guess=[]
        for t in keys:
            labels={label for st,en,label in valid if st<=t<en}
            if len(labels)!=1:continue
            truth.append(next(iter(labels)));guess.append(pred[t]['action_temporal'])
        if not truth:print(name,'sin tiempos revisados coincidentes');continue
        print(name,'aciertos en fotogramas etiquetados:',sum(a==b for a,b in zip(truth,guess)),'/',len(truth))
        for label in ('exploring_sniffing','locomotion','resting'):
            idx=[i for i,v in enumerate(truth) if v==label]
            if idx:print(' ',label,':',sum(guess[i]==label for i in idx),'/',len(idx))
    print('Nota: fotogramas consecutivos correlacionados; no constituyen muestras independientes.')
if __name__=='__main__':main()
