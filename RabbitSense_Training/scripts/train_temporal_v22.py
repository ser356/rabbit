#!/usr/bin/env python3
"""Entrena un modelo de acciones REALES a partir de CLIPS ETIQUETADOS.

La imagen fija ViT no aprende emociones. El modelo temporal solo aprende las
etiquetas que se le den y exige separación por vídeo si hay varias grabaciones.
"""
from __future__ import annotations
import argparse,csv,json,random,time,hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from temporal_v22 import window_features,TemporalNet
ROOT=Path(__file__).resolve().parents[1]

def device_name():
    return 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

def key_for_video(video):
    full=str(Path(video).expanduser().resolve())
    return Path(video).stem+'_'+hashlib.sha256(full.encode()).hexdigest()[:10]

REMAP={'flopping_loafing':'resting','laying':'resting', 'jumping_binkying':'jumping_binkying'}
ALLOWED={'exploring_sniffing','eating','resting','locomotion','digging','grooming',
         'jumping_binkying','periscoping','humping_mating','other','unknown',
         'flopping_loafing','laying'}

def rows_from_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('Sin etiquetas: usa label_v22.command y exporta el CSV')
    for r in rows:
        if r.get('label') not in ALLOWED:raise ValueError(f"Etiqueta no admitida: {r.get('label')}")
        if not r.get('source_video') or not r.get('clip_id'):
            raise ValueError('Cada fila debe tener source_video, clip_id y label')
    return rows

def prepare_examples(label_paths,cache_dir,frames_per_clip):
    all_rows=[]
    for p in label_paths:all_rows+=rows_from_csv(p)
    # El mismo clip no puede tener dos etiquetas contradictorias.
    seen={}; grouped=defaultdict(list)
    for row in all_rows:
        label=REMAP.get(row['label'],row['label'])
        if label=='unknown':continue
        video=str(Path(row['source_video']).expanduser().resolve())
        ident=(video,str(row['clip_id']))
        if ident in seen:
            if seen[ident]!=label:raise ValueError('Etiquetas contradictorias para '+repr(ident))
            continue
        seen[ident]=label
        grouped[video].append((float(row['start_seconds']),float(row['duration_seconds']),label,row['clip_id']))
    examples=[];dimension=None
    for video,items in grouped.items():
        key=key_for_video(video);cache_path=cache_dir/(key+'.npz')
        if not cache_path.exists():
            raise FileNotFoundError(f'Faltan features: {cache_path}\nPrimero ejecuta run_v22.command "{video}"')
        with np.load(cache_path,allow_pickle=False) as cache:
            features=cache['features'];times=cache['time_seconds']
            if str(cache['source_video'].item())!=video:
                raise ValueError('Fuente de cache no coincide: '+str(cache_path))
        dimension=features.shape[1] if dimension is None else dimension
        if dimension!=features.shape[1]:raise ValueError('Los vídeos usan diferentes modelos de características')
        for start,duration,label,cid in items:
            if duration<=0:raise ValueError('Clip sin duración')
            try:feat=window_features(features,times,start,start+min(duration,4.)-float(np.median(np.diff(times))),frames_per_clip)
            except ValueError as e:raise ValueError(f'{key}/{cid}: {e}') from e
            examples.append({'video':video,'start':start,'end':start+duration,'label':label,
                             'clip_id':cid,'features':feat})
    return examples,int(dimension or 0)

def split_examples(examples,seed,allow_prototype):
    videos=sorted({e['video'] for e in examples})
    if len(videos)>=2:
        holdout=videos[-1]
        train=[e for e in examples if e['video']!=holdout]
        val=[e for e in examples if e['video']==holdout]
        status='validation_on_separate_recording_NOT_separate_rabbit'
        warning='Validación por grabación separada; generalización a conejos/escenarios no demostrada.'
    else:
        if not allow_prototype:
            raise RuntimeError('Solo hay un vídeo. Para validar acciones temporalmente hacen falta 2+ vídeos independientes. '
                               'Para una DEMOSTRACIÓN sin validez externa, pasa --allow-single-video-prototype.')
        ordered=sorted(examples,key=lambda e:e['start'])
        nval=max(2,round(len(ordered)*.25))
        train=ordered[:max(0,len(ordered)-nval-1)]
        val=ordered[len(ordered)-nval:]
        status='prototype_same_video_non_independent'
        warning='PROTOTIPO: entrenamiento y validación proceden del mismo vídeo, con un clip de embargo. NO medir generalización.'
    if not train or not val:raise RuntimeError('No hay clips suficientes para separar train/val')
    if len(train)<5:raise RuntimeError(f'Necesito al menos 5 clips train; hay {len(train)}')
    train_labels={e['label'] for e in train}
    if len(train_labels)<2:
        raise RuntimeError('Necesitamos al menos DOS acciones diferentes etiquetadas en TRAIN; etiquetas: '+str(train_labels))
    absent={e['label'] for e in val}-train_labels
    if absent:raise RuntimeError(f'La validación incluye acciones nunca vistas en entrenamiento: {absent}. Añade otras grabaciones o revisa las etiquetas.')
    counts=Counter(e['label'] for e in train)
    if any(c<2 for c in counts.values()):
        raise RuntimeError('Cada acción entrenada debe tener >=2 clips distintos. Recuento: '+str(counts))
    return train,val,status,warning

def make_arrays(examples,classes):
    x=np.stack([e['features'] for e in examples]).astype('float32')
    y=np.array([classes.index(e['label']) for e in examples],dtype='int64')
    return x,y

def eval_model(model,loader,device,n):
    model.eval();truth=[];pred=[];total=0.
    with torch.inference_mode():
        for x,y in loader:
            out=model(x.to(device));loss=nn.functional.cross_entropy(out,y.to(device))
            total+=float(loss)*len(y)
            truth+=y.tolist();pred+=out.argmax(-1).cpu().tolist()
    return {'loss':total/max(1,len(truth)),
            'accuracy':accuracy_score(truth,pred),
            'macro_f1':f1_score(truth,pred,labels=list(range(n)),average='macro',zero_division=0),
            'true':truth,'pred':pred}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',type=Path,nargs='+',required=True)
    p.add_argument('--cache-dir',type=Path,default=ROOT/'runs/features_v22')
    p.add_argument('--out',type=Path,default=ROOT/'runs/temporal_v22')
    p.add_argument('--frames-per-clip',type=int,default=8)
    p.add_argument('--clip-seconds',type=float,default=4.)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--epochs',type=int,default=75)
    p.add_argument('--patience',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--allow-single-video-prototype',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if args.frames_per_clip<3 or args.epochs<1 or args.batch_size<1:p.error('Argumentos inválidos')
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    examples,dim=prepare_examples(args.labels,args.cache_dir,args.frames_per_clip)
    if len(examples)<8:raise RuntimeError(f'Mínimo 8 clips etiquetados (no unknown); hay {len(examples)}')
    train,val,status,warning=split_examples(examples,42,args.allow_single_video_prototype)
    classes=sorted({e['label'] for e in train})
    if 'exploring_sniffing' not in classes:
        print('AVISO: NO se ha etiquetado exploración en entrenamiento. El modelo NO aprenderá esa clase.')
    print('Clases:',classes,'\nTrain:',len(train),'Val:',len(val),'\n',warning)
    xtr,ytr=make_arrays(train,classes);xval,yval=make_arrays(val,classes)
    print('Distribución train:',dict(Counter(e['label'] for e in train)))
    device=device_name()
    model=TemporalNet(dim,len(classes),args.hidden).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.06)
    loss_fn=nn.CrossEntropyLoss()
    loader=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)),
                      batch_size=args.batch_size,shuffle=True)
    val_loader=DataLoader(TensorDataset(torch.from_numpy(xval),torch.from_numpy(yval)),
                          batch_size=args.batch_size)
    args.out.mkdir(parents=True,exist_ok=True)
    target=args.out/'best.pt'
    if target.exists() and not args.overwrite:raise FileExistsError(f'{target} ya existe. Usa --overwrite conscientemente.')
    best=-1.;wait=0;history=[]
    for epoch in range(args.epochs):
        model.train();tot=0.;count=0
        for x,y in loader:
            x=x.to(device);y=y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(x),y)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step();tot+=float(loss.item())*len(y);count+=len(y)
        ev=eval_model(model,val_loader,device,len(classes))
        metrics={'epoch':epoch+1,'train_loss':tot/max(1,count),'val_loss':ev['loss'],
                 'val_accuracy':ev['accuracy'],'val_macro_f1':ev['macro_f1']}
        history.append(metrics)
        if epoch==0 or (epoch+1)%5==0:print(json.dumps(metrics))
        if ev['macro_f1']>best+1e-5:
            best=ev['macro_f1'];wait=0
            torch.save({'model_state':model.state_dict(),'feat_dim':dim,'hidden':args.hidden,
                        'classes':classes,'frames_per_clip':args.frames_per_clip,
                        'clip_seconds':args.clip_seconds,
                        'training_provenance':status,'training_warning':warning,
                        'training_video_count':len({e['video'] for e in train}),
                        'validation_clip_count':len(val)},target)
        else:wait+=1
        if wait>=args.patience:break
    (args.out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8')
    model.load_state_dict(torch.load(target,map_location='cpu',weights_only=True)['model_state'])
    eval_result=eval_model(model,val_loader,device,len(classes))
    report={'classes':classes,'training_counts':dict(Counter(e['label'] for e in train)),
            'validation_counts':dict(Counter(e['label'] for e in val)),
            'validation_macro_f1':eval_result['macro_f1'],
            'validation_accuracy':eval_result['accuracy'],
            'validation_confusion_matrix':confusion_matrix(eval_result['true'],eval_result['pred'],
                labels=list(range(len(classes)))).tolist(),
            'provenance':status,'warning':warning,'no_independent_test':True,
            'epochs_trained':len(history),'checkpoint':str(target)}
    (args.out/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print('\nModelo temporal:',target,'\n',warning)
    print('Las métricas NO son una prueba clínica ni garantía de funcionamiento con Cinna.')

if __name__=='__main__':main()
