#!/usr/bin/env python3
"""Fine-tuning de un ViT preentrenado con 7 clases observables de conejos.

El resultado NO es un estimador validado de estrés, dolor o emociones.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import platform
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm.auto import tqdm

DEFAULT_MODEL = 'vit_small_patch16_224.augreg_in21k_ft_in1k'
MEAN = (0.485,0.456,0.406)
STD = (0.229,0.224,0.225)


def image_loader(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def select_device(require_mps=False):
    if torch.backends.mps.is_available():return torch.device('mps')
    if torch.cuda.is_available():return torch.device('cuda')
    if require_mps and platform.system()=='Darwin' and platform.machine()=='arm64':
        raise RuntimeError('La GPU MPS no está disponible. Comprueba macOS, PyTorch arm64 y que '
                           'Python no esté ejecutándose con Rosetta. Evitamos entrenar horas en CPU.')
    return torch.device('cpu')


def fixed_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def make_dataloaders(root:Path,batch:int,workers:int,seed:int):
    train_t=transforms.Compose([
        transforms.Resize((256,256),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(224,scale=(0.84,1.0),ratio=(0.92,1.08),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.12,contrast=0.12,saturation=0.09,hue=0.015),
        transforms.ToTensor(),transforms.Normalize(MEAN,STD)
    ])
    eval_t=transforms.Compose([
        transforms.Resize((256,256),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(MEAN,STD)
    ])
    kwargs={'loader':image_loader}
    ds={
        'train':datasets.ImageFolder(root/'train',transform=train_t,**kwargs),
        'val':datasets.ImageFolder(root/'val',transform=eval_t,**kwargs),
        'test':datasets.ImageFolder(root/'test',transform=eval_t,**kwargs),
    }
    assert ds['train'].classes==ds['val'].classes==ds['test'].classes, 'Las clases de splits no coinciden.'
    targets=ds['train'].targets
    class_counts=Counter(targets)
    # Reduce desbalance sin sobreponderar agresivamente las clases raras.
    weights=[1.0/np.sqrt(class_counts[i]) for i in targets]
    sampler=WeightedRandomSampler(weights,num_samples=len(weights),replacement=True,
                                  generator=torch.Generator().manual_seed(seed))
    params=dict(batch_size=batch,num_workers=workers,pin_memory=False,
                persistent_workers=(workers>0))
    loaders={
        'train':DataLoader(ds['train'],sampler=sampler,**params),
        'val':DataLoader(ds['val'],shuffle=False,**params),
        'test':DataLoader(ds['test'],shuffle=False,**params),
    }
    print('Clases:', ds['train'].classes)
    print('Imágenes:', {k:len(d) for k,d in ds.items()})
    return ds,loaders


def infer(model,loader,criterion,device,description):
    model.eval(); total=0.;y_true=[];y_pred=[]
    with torch.inference_mode():
        for x,y in tqdm(loader,desc=description,leave=False):
            x=x.to(device);y_dev=y.to(device)
            logits=model(x)
            loss=criterion(logits,y_dev)
            total+=float(loss.item())*len(y)
            pred=logits.argmax(1).cpu().tolist()
            y_true.extend(y.tolist());y_pred.extend(pred)
    return {'loss':total/max(1,len(y_true)),
            'accuracy':accuracy_score(y_true,y_pred),
            'macro_f1':f1_score(y_true,y_pred,average='macro',zero_division=0),
            'balanced_accuracy':balanced_accuracy_score(y_true,y_pred),
            'true':y_true,'pred':y_pred}


def train(args):
    fixed_seed(args.seed)
    device=select_device(args.require_mps_if_apple_silicon)
    print('Dispositivo:',device,'| PyTorch:',torch.__version__)
    ds,dl=make_dataloaders(args.data,args.batch_size,args.workers,args.seed)
    labels=ds['train'].classes
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'best.pt').exists() and not args.overwrite:
        raise RuntimeError(f'{args.out}/best.pt ya existe. Usa --overwrite para reemplazar el entrenamiento anterior.')
    print('Cargando pesos preentrenados de:',args.model)
    model=timm.create_model(args.model,pretrained=True,num_classes=len(labels))
    model.to(device)
    head=model.get_classifier()
    head_ids={id(p) for p in head.parameters()}
    base_params=[p for p in model.parameters() if id(p) not in head_ids]
    head_params=list(head.parameters())
    opt=torch.optim.AdamW([{'params':base_params,'lr':args.lr_backbone},
                           {'params':head_params,'lr':args.lr_head}],weight_decay=0.05)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=1e-7)
    criterion=nn.CrossEntropyLoss(label_smoothing=0.08)
    history=[];best_f1=-1.;patience=0;t0=time.time()
    for epoch in range(args.epochs):
        freeze=epoch<args.freeze_epochs
        for p in base_params:p.requires_grad=not freeze
        model.train()
        accum=args.accumulate
        opt.zero_grad(set_to_none=True)
        running_loss=0.;seen=0
        correct=0
        bar=tqdm(enumerate(dl['train']),total=len(dl['train']),desc=f'Época {epoch+1}/{args.epochs}' + (' [solo cabeza]' if freeze else ''))
        for step,(images,targets) in bar:
            images=images.to(device);targets=targets.to(device)
            output=model(images)
            loss=criterion(output,targets)
            # Normalización correcta también para el último grupo incompleto.
            effective=min(accum, len(dl['train'])-(step//accum)*accum)
            (loss/effective).backward()
            if (step+1)%accum==0 or step==len(dl['train'])-1:
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
                opt.step();opt.zero_grad(set_to_none=True)
            running_loss+=loss.item()*len(targets)
            seen+=len(targets)
            correct+=(output.argmax(1)==targets).sum().item()
            bar.set_postfix(loss=f'{running_loss/seen:.3f}',accuracy=f'{correct/seen:.3f}')
        val=infer(model,dl['val'],criterion,device,'Validación')
        sched.step()
        metrics={'epoch':epoch+1,'train_loss':running_loss/max(1,seen),'train_accuracy':correct/max(1,seen),
                 'val_loss':val['loss'],'val_accuracy':val['accuracy'],'val_macro_f1':val['macro_f1']}
        history.append(metrics)
        print(json.dumps(metrics,ensure_ascii=False))
        (args.out/'history.json').write_text(json.dumps(history,indent=2))
        if val['macro_f1']>best_f1+1e-5:
            best_f1=val['macro_f1'];patience=0
            torch.save({'model_state':model.state_dict(),'model_name':args.model,
                        'classes':labels,'epoch':epoch+1,'val_macro_f1':best_f1,
                        'normalization':{'mean':MEAN,'std':STD},'img_size':224},args.out/'best.pt')
        else:patience+=1
        if patience>=args.patience:
            print(f'Parada temprana tras {args.patience} épocas sin mejorar.');break

    print('Evaluación final sobre el test independiente (una sola vez)...')
    ckpt=torch.load(args.out/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ckpt['model_state']);model.to(device)
    test=infer(model,dl['test'],criterion,device,'Test final')
    report=classification_report(test['true'],test['pred'],target_names=labels,labels=list(range(len(labels))),
                                 output_dict=True,zero_division=0)
    matrix=confusion_matrix(test['true'],test['pred'],labels=list(range(len(labels))))
    result={'model_name':args.model,'device':str(device),'seed':args.seed,'epochs_completed':len(history),
            'training_seconds':round(time.time()-t0,2),'classes':labels,'best_validation_macro_f1':best_f1,
            'test_accuracy':test['accuracy'],'test_macro_f1':test['macro_f1'],
            'test_balanced_accuracy':test['balanced_accuracy'],'classification_report':report,
            'confusion_matrix':matrix.tolist(),'warning':'Comportamientos observables, NO medición validada de estrés.'}
    (args.out/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('\nMétricas TEST:',json.dumps({k:result[k] for k in ('test_accuracy','test_macro_f1','test_balanced_accuracy')},indent=2))
    print('Modelo:',args.out/'best.pt')
    print('Informe:',args.out/'results.json')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('data/processed'))
    p.add_argument('--out',type=Path,default=Path('runs/behaviour'))
    p.add_argument('--model',default=DEFAULT_MODEL)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--accumulate',type=int,default=2)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--freeze-epochs',type=int,default=2)
    p.add_argument('--lr-backbone',type=float,default=2e-5)
    p.add_argument('--lr-head',type=float,default=3e-4)
    p.add_argument('--patience',type=int,default=6)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--require-mps-if-apple-silicon',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    if a.accumulate<1 or a.epochs<1 or a.batch_size<1: p.error('batch-size, epochs, accumulate >= 1')
    train(a)
if __name__=='__main__':main()
