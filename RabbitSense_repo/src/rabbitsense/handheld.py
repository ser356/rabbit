"""Cámara en mano: semillas, anclas SAM2 automáticas, verificación de etiquetas y cierre con puerta de suficiencia.

Las anclas son guías generadas por SAM2, nunca etiquetas. Las etiquetas solo cuentan tras verificación humana
sobre el vídeo original; el cierre no se ejecuta si no hay suficientes clips verificados.
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .closure import CRITERIA
from .motion import sha256

VERIFIED = 'revisado_video_continuo'
MINIMUM_SECONDS = 3.
REQUIRED = {'locomotion': CRITERIA['minimum_locomotion_clips'], 'resting': CRITERIA['minimum_resting_clips']}

SEED_HTML = '''<!doctype html><html lang="es"><meta charset="utf-8"><title>Semillas RabbitSense</title>
<style>body{font:16px system-ui;background:#14171c;color:#eee;max-width:820px;margin:20px auto;padding:0 16px}
canvas{max-width:100%;cursor:crosshair;border:1px solid #888;touch-action:none}button{font-size:16px;padding:9px;margin:6px}</style>
<h2>Una caja por vídeo</h2><p>Arrastra un rectángulo ajustado alrededor de Cinna en el primer fotograma de cada vídeo.
Al terminar el último, pulsa «Descargar semillas».</p><div id="status"></div><canvas id="c"></canvas>
<p><button onclick="move(-1)">Anterior</button><button onclick="move(1)">Siguiente</button><button onclick="save()">Descargar semillas</button></p>
<script>
const items=__ITEMS__;let i=0,start=null,rect=null,boxes={};const imgs=items.map(x=>{const im=new Image();im.src=x.image;return im});
const c=document.getElementById('c'),ctx=c.getContext('2d');
function draw(){const im=imgs[i];if(!im.complete){im.onload=draw;return}c.width=im.naturalWidth;c.height=im.naturalHeight;ctx.drawImage(im,0,0);
const r=rect||boxes[items[i].name];if(r){ctx.strokeStyle='#00ff77';ctx.lineWidth=3;ctx.strokeRect(...r)}
document.getElementById('status').textContent=`${items[i].name} (${i+1}/${items.length}) · marcados ${Object.keys(boxes).length}/${items.length}`}
function pos(e){const b=c.getBoundingClientRect();return[(e.clientX-b.left)*c.width/b.width,(e.clientY-b.top)*c.height/b.height]}
c.onpointerdown=e=>{start=pos(e);rect=null;c.setPointerCapture(e.pointerId)};
c.onpointermove=e=>{if(!start)return;const p=pos(e);rect=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.abs(p[0]-start[0]),Math.abs(p[1]-start[1])];draw()};
c.onpointerup=()=>{if(!start)return;start=null;if(rect&&rect[2]>8&&rect[3]>8){boxes[items[i].name]=rect;rect=null;move(1)}else{rect=null;draw()}};
function move(d){i=Math.max(0,Math.min(items.length-1,i+d));draw()}
function save(){const out={};for(const x of items){if(!boxes[x.name]){alert('Falta '+x.name);return}out[x.name]={width:x.width,height:x.height,box:boxes[x.name]}}
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));a.download='semillas.json';a.click()}
draw();
</script></html>'''

REVIEW_HTML = '''<!doctype html><html lang="es"><meta charset="utf-8"><title>Verificar etiquetas</title>
<style>body{font:15px system-ui;background:#14171c;color:#eee;margin:0;padding:16px}main{display:flex;gap:16px;flex-wrap:wrap}
video{max-height:78vh;max-width:100%;background:#000}ol{max-height:78vh;overflow:auto;min-width:300px;margin:0}
li{padding:3px 6px;cursor:pointer}li.on{background:#2b3a55}.ok{color:#7ee787}.skip{color:#777}
button{font-size:15px;padding:8px 10px;margin:4px}#counts{font-weight:600}</style>
<p>Safari. Cada clip se repite en bucle sobre el vídeo original. Mira el <b>suelo</b>, no la posición en el encuadre.
<b>L</b> = cambia de sitio en algún momento · <b>R</b> = no se desplaza en los 4 s · <b>U</b> = dudoso · ←/→ navegar.
No etiquetes por lo que convenga contar: lo dudoso es U.</p>
<p id="counts"></p><main><video id="v" controls muted playsinline></video><ol id="list"></ol></main>
<p id="now"></p><button onclick="decide('locomotion')">L · locomoción</button><button onclick="decide('resting')">R · reposo</button>
<button onclick="decide('unknown')">U · dudoso</button><button onclick="save()">Descargar CSV verificado</button>
<script>
const clips=__CLIPS__,required=__REQUIRED__,v=document.getElementById('v');let i=clips.findIndex(c=>c.evaluable);if(i<0)i=0;
function label(c){return c.decision||c.label}
function render(){const list=document.getElementById('list');list.innerHTML='';clips.forEach((c,j)=>{const li=document.createElement('li');
li.textContent=`${c.video} ${c.start}–${c.end}s · ${label(c)}${c.decision?' ✓':''}${c.evaluable?'':' (no se evalúa)'}`;
li.className=(j===i?'on ':'')+(c.decision?'ok':c.evaluable?'':'skip');li.onclick=()=>select(j);list.appendChild(li)});
const n=k=>clips.filter(c=>c.evaluable&&c.decision===k).length;
document.getElementById('counts').textContent=`Verificados: locomoción ${n('locomotion')}/${required.locomotion} · reposo ${n('resting')}/${required.resting}`;
const c=clips[i];document.getElementById('now').textContent=`${c.video} ${c.start}–${c.end}s · propuesta: ${c.label}`}
function select(j){i=j;const c=clips[i];if(v.dataset.src!==c.src){v.src=c.src;v.dataset.src=c.src}
const go=()=>{v.currentTime=c.start;v.play()};v.readyState>0?go():v.onloadedmetadata=go;render()}
v.ontimeupdate=()=>{const c=clips[i];if(v.currentTime>=c.end||v.currentTime<c.start-.25){v.currentTime=c.start}};
function decide(k){clips[i].decision=k;let j=i+1;while(j<clips.length&&!clips[j].evaluable)j++;select(j<clips.length?j:i)}
window.onkeydown=e=>{const k=e.key.toLowerCase();if(k==='l')decide('locomotion');if(k==='r')decide('resting');if(k==='u')decide('unknown');
if(e.key==='ArrowRight')select(Math.min(clips.length-1,i+1));if(e.key==='ArrowLeft')select(Math.max(0,i-1))};
function q(s){return '"'+String(s).replaceAll('"','""')+'"'}
function save(){const rows=['video,start_s,end_s,label,review_status,notes'];for(const c of clips){const done=!!c.decision;
rows.push([c.video,c.start,c.end,label(c),done?'__VERIFIED__':c.status,q(done?`Verificado en vídeo original; propuesta previa: ${c.label}`:c.notes)].join(','))}
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')+'\\n'],{type:'text/csv'}));a.download='etiquetas_verificadas.csv';a.click()}
select(i);
</script></html>'''


def video_path(videos, name):
    matches = sorted(p for p in Path(videos).expanduser().glob(f'{name}.*') if p.suffix.lower() in ('.mov', '.mp4'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Se esperaba un único vídeo {name}.MOV/.MP4 en {videos}; encontrados: {matches}')
    return matches[0].resolve()


def seeds(videos, names, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for name in names:
        capture = cv2.VideoCapture(str(video_path(videos, name)))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f'No se puede leer el primer fotograma de {name}.')
        h, w = frame.shape[:2]
        width = min(600, w)
        frame = cv2.resize(frame, (width, round(h * width / w)))
        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        items.append(dict(name=name, width=frame.shape[1], height=frame.shape[0],
                          image='data:image/jpeg;base64,' + base64.b64encode(buffer).decode()))
    page = out / 'semillas.html'
    page.write_text(SEED_HTML.replace('__ITEMS__', json.dumps(items)), encoding='utf8')
    return page


def masks_to_anchors(directory, path, minimum_fraction=.001):
    """Una guía por fotograma desde máscaras SAM2; máscaras vacías o diminutas dejan la guía ausente."""
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    marks, width, height = [], None, None
    for frame in range(summary['frames']):
        mask = cv2.imread(str(directory / 'masks' / f'{frame:06d}.png'), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f'Máscara ausente: {frame}.')
        height, width = mask.shape
        visible = np.count_nonzero(mask) >= minimum_fraction * mask.size
        box = [float(v) for v in cv2.boundingRect(mask)] if visible else None
        marks.append(dict(time=frame / summary['fps'], box=box))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(width=width, height=height, marks=marks, source='sam2_single_seed',
                                          note='Guías generadas por SAM2 desde una caja inicial; no son verdad-terreno.',
                                          segmentation_sha256=sha256(directory / 'summary.json'))))
    return sum(m['box'] is not None for m in marks), len(marks)


def anchors(videos, seeds_path, model, out, device):
    from .segment import segment
    out = Path(out)
    for name, seed in json.loads(Path(seeds_path).read_text()).items():
        seed_path = out / 'seeds' / f'{name}.json'
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_path.write_text(json.dumps(dict(width=seed['width'], height=seed['height'],
                                             marks=[dict(time=0., box=seed['box'])])))
        destination = out / 'segmentation' / name
        segment(video_path(videos, name), seed_path, model, destination, device)
        visible, total = masks_to_anchors(destination, out / 'anchors' / f'{name}.json')
        print(f'ANCLAS {name}: {visible}/{total} fotogramas con máscara', flush=True)


def review(videos, labels, out):
    rows = pd.read_csv(labels).fillna({'notes': ''})
    clips = [dict(video=r.video, start=float(r.start_s), end=float(r.end_s), label=r.label, status=r.review_status,
                  notes=r.notes, evaluable=float(r.end_s) - float(r.start_s) >= MINIMUM_SECONDS,
                  src=video_path(videos, r.video).as_uri()) for r in rows.itertuples()]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    page = out / 'revision.html'
    page.write_text(REVIEW_HTML.replace('__CLIPS__', json.dumps(clips)).replace('__REQUIRED__', json.dumps(REQUIRED))
                    .replace('__VERIFIED__', VERIFIED), encoding='utf8')
    return page


def grid(videos, names, output, length=4.):
    """Clips consecutivos de 4 s sin etiqueta propuesta: se etiquetan desde cero en revision.html."""
    rows = []
    for name in names:
        capture = cv2.VideoCapture(str(video_path(videos, name)))
        fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if not np.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f'Metadatos inválidos: {name}.')
        duration = round(count / fps, 3)
        rows += [dict(video=name, start_s=start, end_s=round(min(start + length, duration), 3), label='unknown',
                      review_status='sin_etiquetar', notes='') for start in np.arange(0., duration, length)]
    pd.DataFrame(rows).to_csv(output, index=False)
    return len(rows)


def build_manifest(videos, names, labels, runs, output):
    runs = Path(runs).resolve()
    cases = [dict(name=name, video=str(video_path(videos, name)), anchors=str(runs / 'anchors' / f'{name}.json'),
                  masks=str(runs / 'segmentation' / name), labels=str(Path(labels).resolve())) for name in names]
    Path(output).write_text(json.dumps(dict(cases=cases), indent=2) + '\n')
    return Path(output)


def prepare(labels, output, extension='.MOV', videos=None):
    """Formato de evaluación: solo clips verificados en vídeo continuo y de al menos 3 s quedan reviewed=si."""
    rows = pd.read_csv(labels).fillna({'notes': ''})
    duration = rows.end_s.astype(float) - rows.start_s.astype(float)
    verified = (rows.review_status == VERIFIED) & (duration >= MINIMUM_SECONDS)
    result = pd.DataFrame(dict(clip_id=[f'mano_{v}_{s:g}' for v, s in zip(rows.video, rows.start_s.astype(float))],
                               video_id=[video_path(videos, v).name if videos else v + extension
                                         for v in rows.video], start_seconds=rows.start_s.astype(float),
                               duration_seconds=duration.round(3), label=rows.label,
                               reviewed=np.where(verified, 'si', 'no'), review_status=rows.review_status,
                               notes=rows.notes))
    result.to_csv(output, index=False)
    return sufficiency(result)


def sufficiency(labels):
    labels = pd.read_csv(labels) if not isinstance(labels, pd.DataFrame) else labels
    evaluable = labels.duration_seconds >= MINIMUM_SECONDS
    report = {}
    for label, required in REQUIRED.items():
        verified = int(((labels.label == label) & (labels.reviewed == 'si') & evaluable).sum())
        pending = int(((labels.label == label) & (labels.reviewed != 'si') & evaluable).sum())
        report[label] = dict(verified=verified, pending_candidates=pending, required=required,
                             reachable_if_all_pending_confirmed=verified + pending >= required)
    report['sufficient'] = all(report[k]['verified'] >= report[k]['required'] for k in REQUIRED)
    report['rule'] = (f'Clips de al menos {MINIMUM_SECONDS:g} s verificados en vídeo continuo. '
                      'Sin suficiencia no se ejecuta el cierre; no se reetiqueta para alcanzar el mínimo.')
    return report


def resolved_manifest(manifest, labels, replacement, path):
    """Copia del manifest con rutas absolutas; los casos que usan `labels` pasan a usar `replacement`."""
    manifest = Path(manifest).resolve()
    cases = json.loads(manifest.read_text())['cases']
    resolve = lambda value: (manifest.parent / Path(value).expanduser()).resolve()
    for case in cases:
        for key in ('video', 'anchors', 'labels', 'tracks'):
            if key in case:
                case[key] = str(resolve(case[key]))
        if Path(case['labels']) == Path(labels).resolve():
            case['labels'] = str(Path(replacement).resolve())
    Path(path).write_text(json.dumps(dict(cases=cases), indent=2))
    return Path(path)


def relabel(path, result, status, report, label_source):
    result['status'] = status
    result['passing_methods'] = []
    for method in result['methods'].values():
        method['meets_thresholds_not_a_verdict'] = method.pop('passed')
    result.update(sufficiency=report, label_source=label_source)
    Path(path).write_text(json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False))
    print(f'\nESTADO REAL: {status} ({label_source}); no es un veredicto de cierre.')
    return result


def run(manifest, labels, out, descriptive=False, include_unverified=False, informative=None):
    report = sufficiency(labels)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if include_unverified and not descriptive:
        raise ValueError('--include-unverified solo se admite en modo descriptivo.')
    if not report['sufficient'] and not descriptive:
        print('NO SE EJECUTA: clips verificados insuficientes. Grabar más; no forzar etiquetas.', file=sys.stderr)
        return None
    from .closure import close
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    used, source = Path(labels), 'verified_continuous_video_only'
    if include_unverified:
        frame = pd.read_csv(labels)
        frame['reviewed'] = np.where(frame.duration_seconds >= MINIMUM_SECONDS, 'si', 'no')
        used, source = out / 'labels_used.csv', 'includes_unverified_proposals'
        frame.to_csv(used, index=False)
    decisive = close(resolved_manifest(manifest, labels, used, out / 'manifest_used.json'), out / 'decisive',
                     announce=report['sufficient'])
    if not report['sufficient']:
        decisive = relabel(out / 'decisive' / 'verdict.json', decisive, 'descriptive_only_insufficient_sample',
                           report, source)
    if informative:
        result = close(resolved_manifest(informative, labels, used, out / 'informative_manifest_used.json'),
                       out / 'informative', announce=False)
        relabel(out / 'informative' / 'verdict.json', result, 'informative_not_decisive', report, source)
    return decisive


def main():
    parser = argparse.ArgumentParser(description='Evaluación de locomoción con cámara en mano')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('seeds', 'anchors', 'review'):
        command = sub.add_parser(name)
        command.add_argument('--videos', required=True, help='Carpeta con los .MOV originales')
        command.add_argument('--out', required=True)
    sub.choices['seeds'].add_argument('--names', nargs='+', required=True)
    sub.choices['anchors'].add_argument('--seeds', required=True)
    sub.choices['anchors'].add_argument('--model', required=True)
    sub.choices['anchors'].add_argument('--device', choices=['cpu', 'mps'], default='mps')
    sub.choices['review'].add_argument('--labels', required=True)
    command = sub.add_parser('prepare')
    command.add_argument('--labels', required=True, help='CSV verificado exportado por revision.html')
    command.add_argument('--output', required=True)
    command.add_argument('--videos', help='Carpeta de los originales, para usar su extensión real')
    command = sub.add_parser('grid')
    for name in ('videos', 'output'):
        command.add_argument(f'--{name}', required=True)
    command.add_argument('--names', nargs='+', required=True)
    command = sub.add_parser('manifest')
    for name in ('videos', 'labels', 'runs', 'output'):
        command.add_argument(f'--{name}', required=True)
    command.add_argument('--names', nargs='+', required=True)
    command = sub.add_parser('run')
    for name in ('manifest', 'labels', 'out'):
        command.add_argument(f'--{name}', required=True)
    command.add_argument('--descriptive', action='store_true',
                         help='Ejecuta aunque falten clips; el resultado nunca es un veredicto de cierre')
    command.add_argument('--include-unverified', action='store_true',
                         help='Solo descriptivo: evalúa también propuestas no verificadas, marcadas como tales')
    command.add_argument('--informative', help='Manifest combinado, siempre informativo')
    args = parser.parse_args()
    if args.command == 'seeds':
        print('Abre:', seeds(args.videos, args.names, args.out))
    elif args.command == 'anchors':
        anchors(args.videos, args.seeds, args.model, args.out, args.device)
    elif args.command == 'review':
        print('Abre en Safari:', review(args.videos, args.labels, args.out))
    elif args.command == 'prepare':
        report = prepare(args.labels, args.output, videos=args.videos)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        sys.exit(0 if report['sufficient'] else 2)
    elif args.command == 'grid':
        print('Clips a etiquetar:', grid(args.videos, args.names, args.output))
    elif args.command == 'manifest':
        print('Manifest:', build_manifest(args.videos, args.names, args.labels, args.runs, args.output))
    elif run(args.manifest, args.labels, args.out, args.descriptive, args.include_unverified,
             args.informative) is None:
        sys.exit(2)


if __name__ == '__main__':
    main()
