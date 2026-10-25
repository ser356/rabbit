#!/usr/bin/env python3
"""Two-class temporal demonstrator: grouped by *recording*, never by frames.

Validation recordings are fixed before fitting:
- IMG_5006.MOV (exploration)
- 972F531F-...mp4 (rest)
They are not a separate-rabbit clinical test set.
"""
from __future__ import annotations
import argparse,json,random
from collections import Counter
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
from temporal_v22 import TemporalNet
from train_temporal_v22 import prepare_examples, make_arrays

ROOT=Path(__file__).resolve().parents[1]
HOLDOUT={'IMG_5006.MOV','972F531F-A923-44DC-B69F-0B536A239FC3.mp4'}
CLASSES=['exploring_sniffing','resting']


def measure(net,loader,device):
    net.eval();total=0.;y=[];p=[]
    with torch.inference_mode():
        for xx,yy in loader:
            z=net(xx.to(device));loss=nn.functional.cross_entropy(z,yy.to(device))
            total+=float(loss)*len(yy);y.extend(yy.tolist());p.extend(z.argmax(-1).cpu().tolist())
    return dict(val_loss=round(total/max(1,len(y)),6),
                val_accuracy=round(float(accuracy_score(y,p)),5),
                val_macro_f1=round(float(f1_score(y,p,labels=[0,1],average='macro',zero_division=0)),5),
                val_balanced_accuracy=round(float(balanced_accuracy_score(y,p)),5),
                confusion_matrix=confusion_matrix(y,p,labels=[0,1]).tolist())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--patience',type=int,default=12)
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    labels=ROOT/'runs/review/cinna_v23_labels.csv'
    if not labels.exists():raise SystemExit('Faltan etiquetas/cachés. Primero ejecuta bash train_multivideo_v23.command')
    examples,dim=prepare_examples([labels],ROOT/'runs/features_v22',frames_per_clip=8)
    if dim<10:raise RuntimeError('Feature dim inválida')
    allvideos={e['video'] for e in examples}
    if len(allvideos)!=7:raise RuntimeError(f'Esperados 7 vídeos útiles; encontrados {len(allvideos)}')
    train=[e for e in examples if Path(e['video']).name not in HOLDOUT]
    val=[e for e in examples if Path(e['video']).name in HOLDOUT]
    if len(train)!=12 or len(val)!=6:raise RuntimeError(f'Split inesperado: train {len(train)}, validación {len(val)}')
    if {e['video'] for e in train}&{e['video'] for e in val}:raise AssertionError('Fuga: vídeo repetido entre train y val')
    if set(e['label'] for e in train)!=set(CLASSES) or set(e['label'] for e in val)!=set(CLASSES):
        raise RuntimeError('Falta una clase en train/validation')
    print('Clases:',CLASSES)
    print('TRAIN:',len(train),'clips de',len(set(e['video'] for e in train)),'vídeos',dict(Counter(e['label'] for e in train)))
    print('VALIDACIÓN:',len(val),'clips de',len(set(e['video'] for e in val)),'vídeos',dict(Counter(e['label'] for e in val)))
    print('IMPORTANTE: no son vídeos de otro conejo, NO es una prueba de generalización clínica.')
    xtr,ytr=make_arrays(train,CLASSES);xval,yval=make_arrays(val,CLASSES)
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    model=TemporalNet(dim,2,hidden=64).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.08)
    dl=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)),batch_size=4,shuffle=True)
    vl=DataLoader(TensorDataset(torch.from_numpy(xval),torch.from_numpy(yval)),batch_size=6)
    dst=ROOT/'runs/temporal_v23';dst.mkdir(parents=True,exist_ok=True)
    best=dst/'best.pt'
    if best.exists() and not args.overwrite:raise FileExistsError(f'El modelo ya existe {best}. No sobrescrito. Pasa --overwrite si quieres reentrenarlo.')
    bestscore=-1.;bestloss=float('inf');early=0;history=[]
    for epoch in range(1,args.epochs+1):
        model.train();losses=[]
        for xx,yy in dl:
            xx=xx.to(device);yy=yy.to(device)
            optim.zero_grad(set_to_none=True)
            logits=model(xx);loss=nn.functional.cross_entropy(logits,yy)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step();losses.append(float(loss.detach().item()))
        metrics=measure(model,vl,device)
        metrics.update(epoch=epoch,train_loss=round(float(np.mean(losses)),6))
        history.append(metrics)
        improved=(metrics['val_macro_f1']>bestscore+1e-5 or
                  (abs(metrics['val_macro_f1']-bestscore)<=1e-5 and metrics['val_loss']<bestloss-1e-5))
        if improved:
            bestscore=metrics['val_macro_f1'];bestloss=metrics['val_loss'];early=0
            torch.save({'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'feat_dim':dim,'hidden':64,
                        'classes':CLASSES,'frames_per_clip':8,'clip_seconds':4.,
                        'training_provenance':'grouped_recording_holdout_same_rabbit_prototype',
                        'training_warning':'Solo 12 clips train y 6 val de Cinna, clases exclusivas explorar/reposar. Validación usada para early stopping, no test independiente. Cajas manuales aproximadas y vídeo comprimido.',
                        'training_video_count':len({e['video'] for e in train}),
                        'validation_video_count':len({e['video'] for e in val}),
                        'validation_clip_count':len(val)},best)
        else:early+=1
        if epoch==1 or epoch%5==0 or early>=args.patience:print(json.dumps(metrics,ensure_ascii=False))
        if early>=args.patience:
            print('Early stopping,',args.patience,'épocas sin mejorar.');break
    report={'classes':CLASSES,'training_clips':len(train),'validation_clips':len(val),
            'training_videos':[str(x) for x in sorted({e['video'] for e in train})],
            'validation_videos':[str(x) for x in sorted({e['video'] for e in val})],
            'best_val_macro_f1':bestscore,'best_val_loss':bestloss,
            'limitations':['no test independiente', 'un solo conejo',
                           'pocos clips y clases', 'anclas aproximadas', 'vídeos comprimidos',
                           'NO estrés ni dolor', 'no apto para decidir cuidados médicos']}
    (dst/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    (dst/'history.json').write_text(json.dumps(history,indent=2,ensure_ascii=False))
    print('\nMODELO NUEVO:',best,'\nINFORME:',dst/'results.json')
    print('El anterior runs/temporal_v22/best.pt no se ha modificado.')

if __name__=='__main__':main()
