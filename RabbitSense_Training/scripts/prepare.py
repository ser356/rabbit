#!/usr/bin/env python3
"""Descarga el Dataset C del paper IEEE y crea splits sin fugas por imágenes casi duplicadas.

No genera etiquetas artificiales: solo utiliza nombres reales de clases en carpetas/archivos.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DRIVE_C = 'https://drive.google.com/drive/folders/1Hqig4HIts4FsGy2ubDEEu7g18ci2XmwU'
DRIVE_A = 'https://drive.google.com/drive/folders/1C3NMJ24akHAj2SDKdQ_rPZtogUAKOcjO'
CLASSES = [
    'digging', 'humping_mating', 'eating', 'periscoping',
    'jumping_binkying', 'laying', 'flopping_loafing'
]
ALIASES = {
    'digging': [r'digg?(?:ing)?', r'cavando'],
    'humping_mating': [r'hump(?:ing)?', r'mat(?:e|ing)', r'monta'],
    'eating': [r'eat(?:ing)?', r'feed(?:ing)?', r'comiendo'],
    'periscoping': [r'periscop(?:e|ing)?', r'rear(?:ing)?', r'erguido'],
    'jumping_binkying': [r'jump(?:ing)?', r'bink(?:y|ying|ies)', r'salt(?:o|ando)'],
    'laying': [r'lay(?:ing)?', r'lying', r'tumbad[oa]'],
    'flopping_loafing': [r'flop(?:ping)?', r'loaf(?:ing)?', r'escampad[oa]'],
}
EXTS = {'.jpg','.jpeg','.png','.bmp','.webp','.tiff','.tif'}


def clean(s: str) -> str:
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')
    return s


def infer_label(path: Path, source: Path) -> str | None:
    """Cercanía a la imagen primero; evita etiquetar por el nombre genérico del dataset."""
    relative = path.relative_to(source)
    candidates = list(reversed(relative.parts[:-1])) + [path.stem]
    for name in candidates:
        s = clean(name)
        hits = []
        for label, patterns in ALIASES.items():
            if any(re.search(r'(?:^|_)'+p+r'(?:_|$|\d)',s) for p in patterns):
                hits.append(label)
        if len(hits)==1:
            return hits[0]
    return None


def dhash(path: Path) -> int:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        pix = list(im.convert('L').resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value=0
    for row in range(8):
        for col in range(8):
            value = (value<<1) | (pix[row*9+col] > pix[row*9+col+1])
    return value


def safe_extract_zips(root: Path):
    for archive in list(root.rglob('*.zip')):
        destination = archive.parent / ('_unpacked_' + archive.stem)
        if destination.exists():
            continue
        print('Descomprimiendo:',archive.name)
        destination.mkdir(parents=True)
        with zipfile.ZipFile(archive) as z:
            for f in z.infolist():
                target = (destination/f.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise RuntimeError(f'Ruta insegura en ZIP: {f.filename}')
                if (f.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError(f'Enlace simbólico en ZIP: {f.filename}')
            z.extractall(destination)


def local_inventory(dest: Path):
    """Conteo rápido de fotos identificables; el análisis completo se hace en build()."""
    counts = Counter()
    if dest.is_dir():
        for p in dest.rglob('*'):
            if p.is_file() and p.suffix.lower() in EXTS and p.stat().st_size:
                cls = infer_label(p, dest)
                if cls:
                    counts[cls] += 1
    return counts


def enough_images(counts):
    # Umbrales mínimos de la preparación. No significan que la descarga esté completa.
    return sum(counts.values()) >= 300 and all(counts.get(c, 0) >= 12 for c in CLASSES)


def rclone_download(url: str, dest: Path):
    # Requiere una conexión de Google Drive autorizada previamente por el usuario.
    from urllib.parse import urlparse
    folder_id = urlparse(url).path.rstrip('/').split('/')[-1]
    remote = os.environ.get('RABBITSENSE_DRIVE_REMOTE', '').strip().rstrip(':')
    if not remote:
        raise RuntimeError('Falta RABBITSENSE_DRIVE_REMOTE. Ejecuta download_rclone.command para configurarlo.')
    if shutil.which('rclone') is None:
        raise RuntimeError('No está instalado rclone. Ejecuta download_rclone.command.')
    print(f'Descargando con rclone desde carpeta {folder_id}, remoto {remote}:', flush=True)
    cmd = ['rclone', 'copy', remote+':', str(dest), '--drive-root-folder-id', folder_id,
           '--transfers', '2', '--checkers', '2', '--tpslimit', '3',
           '--retries', '3', '--low-level-retries', '5', '--progress']
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def ensure_drive_folder(url: str, dest: Path, method='auto', ignore_errors=False):
    if dest.is_dir() and any(dest.rglob('*.zip')):
        print('Usando ZIP locales descargados desde el navegador...',flush=True)
        safe_extract_zips(dest)
    counts = local_inventory(dest)
    print('Imágenes locales por clase:', dict(counts), flush=True)
    if enough_images(counts):
        print(f'Hay {sum(counts.values())} imágenes etiquetables. Se conservan y se continúa sin descargarlas de nuevo.')
        if sum(counts.values()) < 900:
            print('AVISO: son menos de las ~1100 del estudio: conviene completar el dataset antes de evaluar el modelo.')
        return
    if method == 'skip':
        return
    dest.mkdir(parents=True, exist_ok=True)
    if method == 'rclone' or (method == 'auto' and os.environ.get('RABBITSENSE_DRIVE_REMOTE')):
        success = rclone_download(url, dest)
    else:
        # gdown puede ser limitado por Google incluso si el enlace de carpeta es público.
        print('Intentando gdown (si Drive limita descargas, se facilitará la alternativa):', url, flush=True)
        cmd = [sys.executable, '-m', 'gdown', url, '-O', str(dest), '--continue', '--retries', '1']
        success = subprocess.run(cmd, check=False).returncode == 0
    counts = local_inventory(dest)
    print('Tras la descarga, imágenes por clase:', dict(counts), flush=True)
    if not success and not enough_images(counts):
        if ignore_errors:
            print('AVISO: no se pudo completar la descarga opcional.')
            return
        raise RuntimeError(
            'Google Drive ha rechazado o limitado descargas y aún faltan imágenes por clase. '
            'No se ha iniciado el entrenamiento para evitar resultados erróneos.\n'
            f'CARPETA ORIGINAL: {url}\n'
            'SOLUCIÓN 1: abre la carpeta en el navegador, selecciona todo, Descargar; '
            f'copia los ZIP obtenidos sin extraer a {dest}, y ejecuta bash start.command --no-download.\n'
            'SOLUCIÓN 2: ejecuta bash download_rclone.command para descargar por Google Drive autenticado.'
        )
    if not success:
        print('AVISO: gdown falló, pero hay material local suficiente para entrenar. '
              'Podría ser un subconjunto incompleto; revisa las métricas y procedencia.')


def scan(root: Path):
    items=[]; bad=[]; unlabelled=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        label = infer_label(p,root)
        if label is None:
            unlabelled.append(str(p.relative_to(root)))
            continue
        try:
            h=dhash(p)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            bad.append(f'{p}: {e}');continue
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        items.append({'path':p,'label':label,'hash':h,'sha':sha})
    return items, bad, unlabelled


class UnionFind:
    def __init__(self,n): self.p=list(range(n));self.rank=[0]*n
    def root(self,i):
        while i!=self.p[i]:
            self.p[i]=self.p[self.p[i]]
            i=self.p[i]
        return i
    def join(self,i,j):
        a,b=self.root(i),self.root(j)
        if a==b:return
        if self.rank[a]<self.rank[b]: a,b=b,a
        self.p[b]=a
        if self.rank[a]==self.rank[b]: self.rank[a]+=1


def dedup_and_groups(items, distance=2):
    # Mismo archivo en categorías diferentes: contradicción, se excluyen todos esos archivos.
    by_sha=defaultdict(list)
    for i,r in enumerate(items):by_sha[r['sha']].append(i)
    conflict_sha={key for key, ids in by_sha.items() if len({items[i]['label'] for i in ids})>1}
    if conflict_sha:
        print(f'AVISO: {len(conflict_sha)} imágenes idénticas con etiquetas contradictorias; excluidas.')
    found=set(); clean_items=[]
    for r in items:
        if r['sha'] in conflict_sha or (r['label'],r['sha']) in found: continue
        found.add((r['label'],r['sha']))
        clean_items.append(r)
    # Agrupar imágenes muy próximas por dHash para que no caigan en splits diferentes.
    uf=UnionFind(len(clean_items))
    for i in range(len(clean_items)):
        for j in range(i):
            if (clean_items[i]['hash'] ^ clean_items[j]['hash']).bit_count() <= distance:
                uf.join(i,j)
    for i,r in enumerate(clean_items):r['group']=str(uf.root(i))
    return clean_items


def split_group_stratified(items,seed):
    import numpy as np
    ys=np.array([CLASSES.index(r['label']) for r in items])
    groups=np.array([r['group'] for r in items])
    unique_groups=len(set(groups.tolist()))
    print('Grupos visualmente independientes (aprox.):',unique_groups)
    if unique_groups < 21:
        raise RuntimeError('No hay suficientes grupos visualmente independientes para dividir en train/val/test.')
    splitter=StratifiedGroupKFold(n_splits=7,shuffle=True,random_state=seed)
    indices=np.arange(len(items))
    trainval_idx,test_idx=next(splitter.split(indices,ys,groups))
    splitter2=StratifiedGroupKFold(n_splits=6,shuffle=True,random_state=seed+1)
    reltrain,relval=next(splitter2.split(trainval_idx,ys[trainval_idx],groups[trainval_idx]))
    train_idx,val_idx=trainval_idx[reltrain],trainval_idx[relval]
    partitions={'train':train_idx,'val':val_idx,'test':test_idx}
    seen={part:set(groups[ids].tolist()) for part,ids in partitions.items()}
    assert not seen['train']&seen['val'] and not seen['train']&seen['test'] and not seen['val']&seen['test']
    return partitions


def build(src, out, allow_small=False,seed=42):
    if out.exists() and (out/'manifest.csv').exists():
        try:
            state=json.loads((out/'stats.json').read_text())
            if state.get('source')==str(src.resolve()) and state.get('ready'):
                print('Dataset ya preparado:',out,'\nEstadísticas:',state['counts'])
                return
        except Exception: pass
    safe_extract_zips(src)
    items,bad,unlabelled=scan(src)
    counts=Counter(r['label'] for r in items)
    print('Etiquetas detectadas:',dict(counts))
    if unlabelled:
        print(f'Imágenes sin etiqueta: {len(unlabelled)}. Ejemplos:',unlabelled[:8])
    if bad: print(f'Imágenes ilegibles: {len(bad)}. Ejemplos:',bad[:3])
    min_class=2 if allow_small else 12
    min_total=50 if allow_small else 300
    if len(items)<min_total or any(counts.get(cls,0)<min_class for cls in CLASSES):
        dirs=[str(p.relative_to(src)) for p in src.rglob('*') if p.is_dir()]
        raise RuntimeError('Dataset incompleto o no organizado por etiquetas. Se necesitan las 7 clases '
                           f'y al menos {min_class} imágenes/clase ({min_total} en total). '
                           f'Encontrado: {dict(counts)}. Carpetas (muestra): {dirs[:30]}')
    items=dedup_and_groups(items)
    partitions=split_group_stratified(items,seed)
    for part,idxs in partitions.items():
        missing=[label for label in CLASSES if not any(items[int(i)]['label']==label for i in idxs)]
        if missing:
            raise RuntimeError(f'Split {part} sin clases {missing}. Agrupación excesiva o dataset insuficiente. '
                               'Es preferible corregir los splits antes de entrenar.')
    if out.exists():shutil.rmtree(out)
    out.mkdir(parents=True)
    lines=[];split_counts={}
    for split,idxs in partitions.items():
        cc=Counter()
        for k in idxs:
            r=items[int(k)]
            p=r['path']; cls=r['label']
            dest=out/split/cls/(r['sha'][:16]+'_'+hashlib.sha1(str(p).encode()).hexdigest()[:8]+p.suffix.lower())
            dest.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(p,dest) # sin ocupar almacenamiento adicional
            except OSError:shutil.copy2(p,dest)
            lines.append({'split':split,'class':cls,'path':str(p.resolve()),'dataset_path':str(dest.resolve()),'group':r['group']})
            cc[cls]+=1
        split_counts[split]=dict(cc)
    with (out/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['split','class','path','dataset_path','group']);writer.writeheader();writer.writerows(lines)
    state={'source':str(src.resolve()),'ready':True,'counts':split_counts,'classes':CLASSES,'total':len(items),
           'duplicates_removed':sum(counts.values())-len(items),'unlabelled':len(unlabelled),'bad_images':len(bad),'seed':seed}
    (out/'stats.json').write_text(json.dumps(state,indent=2,ensure_ascii=False))
    print('Splits preparados:',json.dumps(split_counts,indent=2,ensure_ascii=False))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=ROOT/'data/raw/dataset_c',help='Carpeta local del dataset C; se descargará si está vacía')
    ap.add_argument('--out',type=Path,default=ROOT/'data/processed')
    ap.add_argument('--download-a',action='store_true',help='Descarga también Dataset A (no usado en este entrenamiento)')
    ap.add_argument('--allow-small',action='store_true',help='Solo para pruebas: reduce el mínimo de muestras')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--no-download',action='store_true',help='Solo usa las imágenes ya guardadas en --source')
    ap.add_argument('--download-method',choices=['auto','gdown','rclone','skip'],default='auto',help='Descarga con gdown (público), rclone (Google autenticado) o sin descarga')
    args=ap.parse_args()
    args.source=args.source.expanduser().resolve()
    args.out=args.out.expanduser().resolve()
    if not args.no_download:ensure_drive_folder(DRIVE_C,args.source,method=args.download_method)
    if args.download_a:ensure_drive_folder(DRIVE_A,ROOT/'data/raw/dataset_a',method=args.download_method,ignore_errors=True)
    build(args.source,args.out,allow_small=args.allow_small,seed=args.seed)

if __name__=='__main__':main()
