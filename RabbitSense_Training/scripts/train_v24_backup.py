#!/usr/bin/env python3
"""Three-class supervised V2.4 training with grouped recording holdout.
Requires manually reviewed locomotion from >=2 different recordings.
"""
import argparse,csv,json,random
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,balanced_accuracy_score
from train_temporal_v22 import prepare_examples,make_arrays
from temporal_v22 import TemporalNet
ROOT=Path(__file__).resolve().parents[1]
CLASSES=['exploring_sniffing','locomotion','resting']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=70);ap.add_argument('--patience',type=int,default=12);ap.add_argument('--overwrite',action='store_true');args=ap.parse_args()
    src=ROOT/'runs/review_v24/etiquetas_para_revisar.csv'
    if not src.exists():raise SystemExit('Primero ejecuta prepare_v24.py y revisa etiquetas_para_revisar.csv')
    with src.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    approved=[]
    for r in rows:
        if r['reviewed'].strip().lower() not in ('si','sí','yes','true'):continue
        if r['label'] not in CLASSES:continue
        if Path(r['source_video']).name.lower()=='cinna.mov':raise SystemExit('FUGA: cinna.mov está reservado para evaluación')
        approved.append(r)
    if not approved:raise SystemExit('No hay clips revisados y etiquetados; NO se entrenará con etiquetas inventadas.')
    grouped=defaultdict(list)
    for r in approved:grouped[str(Path(r['source_video']).resolve())].append(r)
    class_videos={cl:{v for v,rs in grouped.items() if any(r['label']==cl for r in rs)} for cl in CLASSES}
    for cl in CLASSES:
        if len(class_videos[cl])<2:raise SystemExit(f'Faltan ejemplos de {cl} en >=2 grabaciones diferentes. Grabaciones: {len(class_videos[cl])}. No se entrena.')
    # Select one complete validation recording containing all three classes, if available.
    candidates=[v for v,rs in grouped.items() if {r['label'] for r in rs}==set(CLASSES)]
    if not candidates:raise SystemExit('Necesitas al menos un vídeo de validación con ejemplos revisados de las tres clases.')
    holdout=sorted(candidates)[-1]
    train_rows=[r for r in approved if str(Path(r['source_video']).resolve())!=holdout]
    val_rows=[r for r in approved if str(Path(r['source_video']).resolve())==holdout]
    counts=Counter(r['label'] for r in train_rows)
    if any(counts[c]<3 for c in CLASSES):raise SystemExit(f'Necesitas >=3 clips por clase en train, en vídeos diferentes al de validación: {counts}')
    # Use a temporary *reviewed-only* labels CSV; never touch V2.3 labels.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='rabbitsense_v24_') as td:
        temp=Path(td)/'reviewed.csv'
        with temp.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['clip_id','source_video','video_id','start_seconds','duration_seconds','clip_path','label']);w.writeheader()
            for r in approved:w.writerow({k:r.get(k,'') for k in w.fieldnames})
        examples,dim=prepare_examples([temp],ROOT/'runs/features_v22',frames_per_clip=8)
    tr=[e for e in examples if e['video']!=holdout];va=[e for e in examples if e['video']==holdout]
    if not tr or not va:raise SystemExit('Split vacío')
    xtr,ytr=make_arrays(tr,CLASSES);xv,yv=make_arrays(va,CLASSES)
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    net=TemporalNet(dim,3,hidden=64).to(device)
    optim=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.08)
    dl=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)),batch_size=4,shuffle=True)
    vl=DataLoader(TensorDataset(torch.from_numpy(xv),torch.from_numpy(yv)),batch_size=8)
    dst=ROOT/'runs/temporal_v24';dst.mkdir(parents=True,exist_ok=True)
    best=dst/'best.pt'
    if best.exists() and not args.overwrite:raise SystemExit(f'{best} existe; no se sobrescribe sin --overwrite')
    bestf=-1.;bestloss=float('inf');pat=0;history=[]
    for epoch in range(1,args.epochs+1):
        net.train();losses=[]
        for x,y in dl:
            x=x.to(device);y=y.to(device);optim.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(net(x),y);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);optim.step();losses.append(float(loss.detach()))
        net.eval();yy=[];pp=[];losses_val=[]
        with torch.inference_mode():
            for x,y in vl:
                z=net(x.to(device));losses_val.append((float(nn.functional.cross_entropy(z,y.to(device))),len(y)))
                yy+=y.tolist();pp+=z.argmax(-1).cpu().tolist()
        vloss=sum(l*n for l,n in losses_val)/len(yy)
        score=float(f1_score(yy,pp,labels=[0,1,2],average='macro',zero_division=0))
        entry={'epoch':epoch,'train_loss':float(np.mean(losses)),'val_loss':vloss,'val_accuracy':float(accuracy_score(yy,pp)),'val_macro_f1':score,'confusion_matrix':confusion_matrix(yy,pp,labels=[0,1,2]).tolist()}
        history.append(entry)
        if epoch==1 or epoch%5==0:print(json.dumps(entry,ensure_ascii=False))
        if score>bestf+1e-6 or (abs(score-bestf)<1e-6 and vloss<bestloss-1e-6):
            bestf=score;bestloss=vloss;pat=0
            torch.save({'model_state':{k:v.detach().cpu() for k,v in net.state_dict().items()},'feat_dim':dim,'hidden':64,'classes':CLASSES,'frames_per_clip':8,'clip_seconds':4.,'training_provenance':'manual_reviewed_v24_recording_holdout_same_rabbit','training_warning':'No test independiente. Cinna.mov reservado para comparación. No mide estrés/dolor.'},best)
        else:pat+=1
        if pat>=args.patience:break
    report={'classes':CLASSES,'training_clips':len(tr),'validation_clips':len(va),'training_videos':sorted({e['video'] for e in tr}),'validation_video':holdout,'best_val_macro_f1':bestf,'best_val_loss':bestloss,'limitations':['un solo conejo','cajas aproximadas','validación usada para selección de modelo','cinna.mov no usado para entrenar','no inferir estrés, dolor ni bienestar']}
    (dst/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(dst/'history.json').write_text(json.dumps(history,indent=2))
    print('Modelo V2.4:',best)
if __name__=='__main__':main()
