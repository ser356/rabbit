#!/usr/bin/env python3
"""RabbitSense V2.5: small-data baseline with recording-grouped evaluation.
Does not modify V2.2/V2.3/V2.4. Does not train on cinna.mov.
"""
import argparse,csv,json,sys,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from joblib import dump,load
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_temporal_v22 import prepare_examples
CLASSES=['exploring_sniffing','locomotion','resting']

def vec(f):
    f=np.asarray(f,dtype=np.float64)
    if f.ndim!=2:raise ValueError('Se esperaba una matriz de features por clip')
    d=np.diff(f,axis=0)
    return np.concatenate([f.mean(axis=0),f.std(axis=0),np.mean(np.abs(d),axis=0) if len(d) else np.zeros(f.shape[1])]).astype('float32')

def model(n):
    # Low-dimensional regularized baseline to reduce memorization on tiny datasets.
    return Pipeline([('scale',StandardScaler()),('pca',PCA(n_components=min(6,n-1),svd_solver='full')),
                     ('clf',LogisticRegression(C=0.1,class_weight='balanced',max_iter=2000))])

def read_data():
    src=ROOT/'runs/review_v24/etiquetas_para_revisar.csv'
    if not src.exists():raise SystemExit('Falta '+str(src))
    with src.open(encoding='utf-8-sig',newline='') as fh:rows=list(csv.DictReader(fh))
    selected=[]
    for r in rows:
        if r.get('reviewed','').strip().lower() not in ('si','sí','yes','true'):continue
        if r.get('label') not in CLASSES:continue
        if Path(r['source_video']).name.lower()=='cinna.mov':raise SystemExit('cinna.mov debe quedar fuera del entrenamiento')
        selected.append(r)
    if not selected:raise SystemExit('No hay etiquetas revisadas')
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'labels.csv'
        fields=['clip_id','source_video','video_id','start_seconds','duration_seconds','clip_path','label']
        with p.open('w',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for r in selected:w.writerow({k:r.get(k,'') for k in fields})
        ex,dim=prepare_examples([p],ROOT/'runs/features_v22',frames_per_clip=8)
    if len(ex)<8:raise SystemExit('Muy pocos clips para el experimento')
    x=np.stack([vec(e['features']) for e in ex]);y=np.array([e['label'] for e in ex]);g=np.array([e['video'] for e in ex]);return ex,x,y,g,dim

def metrics(y,p):
    return {'accuracy':round(float(accuracy_score(y,p)),4), 'balanced_accuracy':round(float(balanced_accuracy_score(y,p)),4),
            'macro_f1':round(float(f1_score(y,p,labels=CLASSES,average='macro',zero_division=0)),4),
            'confusion_matrix':confusion_matrix(y,p,labels=CLASSES).tolist()}

def train(args):
    ex,x,y,g,dim=read_data();print('Clips:',len(y),'Clases:',dict(Counter(y)),'Vídeos:',len(set(g)))
    if len(set(y))!=3:raise SystemExit('Falta una de las tres clases en los ejemplos revisados')
    out=ROOT/'runs/temporal_v25';out.mkdir(parents=True,exist_ok=True)
    logo=LeaveOneGroupOut();truth=[];pred=[];baseline=[];folds=[]
    for tr,te in logo.split(x,y,g):
        present=set(y[tr]);missing=set(CLASSES)-present
        if missing:
            folds.append({'video':Path(g[te[0]]).name,'status':'no_evaluable_clase_ausente_en_train','missing':sorted(missing)})
            continue
        m=model(len(tr));m.fit(x[tr],y[tr]);p=m.predict(x[te]);b=DummyClassifier(strategy='most_frequent').fit(x[tr],y[tr]).predict(x[te])
        truth.extend(y[te].tolist());pred.extend(p.tolist());baseline.extend(b.tolist())
        folds.append({'video':Path(g[te[0]]).name,'n':len(te),'true':y[te].tolist(),'predicted':p.tolist()})
    report={'classes':CLASSES,'n_clips':len(y),'n_recordings':len(set(g)),'class_counts':dict(Counter(y)),
            'method':'regularized_PCA_logistic_regression_leave_one_recording_out',
            'folds':folds,'n_evaluated':len(truth),'limitations':['mismo conejo','muy pocos ejemplos','no es detector validado de bienestar/estres/dolor','algunos folds no evaluables si falta una clase']}
    if truth:
        report['cross_validation']=metrics(truth,pred)
        report['majority_baseline']=metrics(truth,baseline)
        print('Validación agrupada:',report['cross_validation'])
        print('Baseline mayoría:',report['majority_baseline'])
    else:print('ADVERTENCIA: ninguna partición evaluable por clases ausentes.')
    (out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    # Fit on ALL reviewed recordings only AFTER cross-validation; cinna.mov remains untouched.
    m=model(len(y));m.fit(x,y)
    dump({'pipeline':m,'classes':CLASSES,'feat_dim':dim,'feature_vector':'mean_std_absdiff','n_train':len(y)},out/'baseline.joblib')
    print('Guardado:',out/'baseline.joblib');print('Informe:',out/'evaluation.json')

def test(args):
    obj=load(ROOT/'runs/temporal_v25/baseline.joblib');m=obj['pipeline']
    cache=ROOT/'runs/features_v22';matches=[]
    for p in cache.glob('*.npz'):
        try:
            with np.load(p,allow_pickle=False) as z:
                if Path(str(z['source_video'].item())).name.lower()=='cinna.mov':matches.append(p)
        except (KeyError,ValueError):continue
    if not matches:raise SystemExit('No encuentro features de cinna.mov. Procesa primero cinna.mov con predict_v22.py')
    if len(matches)>1:print('Aviso: varias cachés cinna.mov; usando',matches[0])
    with np.load(matches[0],allow_pickle=False) as z:features=z['features'];times=z['time_seconds']
    if features.shape[1]!=obj['feat_dim']:raise SystemExit('Dimensión incompatible con el modelo')
    out=ROOT/'runs/predictions_v25/cinna';out.mkdir(parents=True,exist_ok=True)
    # Non-overlapping windows, 4 seconds. Same 8-frame temporal sampling as training.
    rows=[]
    for start in np.arange(float(times[0]),float(times[-1]),4.0):
        ids=np.where((times>=start)&(times<start+4.0))[0]
        if len(ids)<3:continue
        positions=np.linspace(0,len(ids)-1,8).round().astype(int)
        x=vec(features[ids[positions]])[None,:]
        prob=m.predict_proba(x)[0];order=np.argsort(prob)[::-1];top=int(order[0]);second=int(order[1]);conf=float(prob[top]);margin=float(prob[top]-prob[second])
        # Heuristic abstention, NOT calibrated confidence.
        label=str(m.classes_[top]) if conf>=0.60 and margin>=0.15 else 'unknown'
        rows.append({'start_seconds':round(float(start),2),'end_seconds':round(float(min(start+4,times[-1])),2),
                     'prediction':label,'candidate':str(m.classes_[top]),'score_uncalibrated':round(conf,4),'margin':round(margin,4),
                     **{'score_'+str(cl):round(float(prob[list(m.classes_).index(cl)]),4) for cl in CLASSES}})
    if not rows:raise SystemExit('No hay ventanas suficientes en las features de cinna.mov')
    with (out/'cinna_v25_windows.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Predicciones:',dict(Counter(r['prediction'] for r in rows)))
    print('CSV:',out/'cinna_v25_windows.csv')
    print('AVISO: scores no calibrados; unknown por umbrales heurísticos. No evalúa dolor/estrés.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['train','cinna']);args=ap.parse_args()
    if args.command=='train':train(args)
    else:test(args)
if __name__=='__main__':main()
