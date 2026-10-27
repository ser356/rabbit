"""Reetiquetado a ciegas de los clips verificados con una definición física de locomoción (fijada el 2026-10-27).

`page`: página con los vídeos originales, sin superposiciones, sin etiquetas previas ni notas, en orden aleatorio y
con códigos opacos. El mapa código → clip se guarda aparte. `apply`: convierte las respuestas en un CSV de etiquetas
con el mismo formato que los existentes, un manifest que lo usa y la concordancia con las etiquetas anteriores.
Las etiquetas anteriores no se modifican.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .motion import sha256

REPO = Path(__file__).resolve().parents[2]
DEFINITION = dict(
    locomotion='Cinna cambia de sitio sobre el suelo con el cuerpo entero, aunque sea despacio.',
    resting='El cuerpo se queda en el mismo sitio. Girar la cabeza, acicalarse o estirarse sin cambiar de sitio es reposo.',
    unknown='No se ve bien, o se recoloca menos de medio cuerpo.')
REVIEWED = {'si', 'sí', 'yes', 'true'}
CLASSES = ('locomotion', 'resting')

PAGE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Reetiquetado ciego</title>
<style>
body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#111;color:#eee}
header{padding:12px 16px;background:#1d1d1d;font-size:14px;line-height:1.5}
main{display:flex;flex-direction:column;align-items:center;padding:12px}
video{max-width:96vw;max-height:68vh;background:#000}
.keys{margin:10px 0;display:flex;gap:8px;flex-wrap:wrap;justify-content:center}
button{font-size:16px;padding:8px 14px;border-radius:6px;border:1px solid #555;background:#2a2a2a;color:#eee;cursor:pointer}
button.on{background:#3a6;border-color:#3a6}
#status{font-size:14px;color:#aaa}
</style></head><body>
<header><b>Definición</b> (no mires etiquetas ni predicciones anteriores):<br>
<b>L · locomoción</b>: __LOC__<br><b>R · reposo</b>: __REST__<br><b>U · desconocido</b>: __UNK__<br>
Espacio: repetir clip · ← →: anterior / siguiente · Las respuestas se guardan en este navegador.</header>
<main>
<div id="status"></div>
<video id="v" muted playsinline></video>
<div class="keys">
<button data-l="locomotion">L · locomoción</button><button data-l="resting">R · reposo</button>
<button data-l="unknown">U · desconocido</button><button id="replay">Repetir</button>
<button id="prev">←</button><button id="next">→</button><button id="save">Descargar CSV</button>
</div></main>
<script>
const clips = __CLIPS__;
const key = 'rabbitsense_ciego_' + __SEED__;
let answers = {};
try { answers = JSON.parse(localStorage.getItem(key) || '{}'); } catch (e) {}
let i = clips.findIndex(c => !answers[c.code]); if (i < 0) i = 0;
const v = document.getElementById('v');
function show() {
  const c = clips[i];
  if (v.dataset.src !== c.src) { v.src = c.src; v.dataset.src = c.src; }
  v.currentTime = c.start; v.play();
  const done = clips.filter(x => answers[x.code]).length;
  document.getElementById('status').textContent = `Clip ${i + 1} de ${clips.length} · respondidos ${done}` +
    (answers[c.code] ? ` · tu respuesta: ${answers[c.code]}` : '');
  document.querySelectorAll('button[data-l]').forEach(b => b.classList.toggle('on', b.dataset.l === answers[c.code]));
}
v.addEventListener('loadedmetadata', () => { v.currentTime = clips[i].start; v.play(); });
v.addEventListener('timeupdate', () => { if (v.currentTime >= clips[i].end) v.currentTime = clips[i].start; });
function answer(label) {
  answers[clips[i].code] = label;
  try { localStorage.setItem(key, JSON.stringify(answers)); } catch (e) {}
  if (i < clips.length - 1) i++;
  show();
}
function save() {
  const rows = ['code,label'].concat(clips.filter(c => answers[c.code]).map(c => `${c.code},${answers[c.code]}`));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([rows.join('\\n') + '\\n'], {type: 'text/csv'}));
  a.download = 'respuestas_ciego.csv'; a.click();
}
document.querySelectorAll('button[data-l]').forEach(b => b.onclick = () => answer(b.dataset.l));
document.getElementById('replay').onclick = show;
document.getElementById('prev').onclick = () => { i = Math.max(0, i - 1); show(); };
document.getElementById('next').onclick = () => { i = Math.min(clips.length - 1, i + 1); show(); };
document.getElementById('save').onclick = save;
document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (k === 'l') answer('locomotion'); else if (k === 'r') answer('resting'); else if (k === 'u') answer('unknown');
  else if (k === ' ') { e.preventDefault(); show(); }
  else if (k === 'arrowleft') { i = Math.max(0, i - 1); show(); }
  else if (k === 'arrowright') { i = Math.min(clips.length - 1, i + 1); show(); }
});
show();
</script></body></html>
"""


def verified_clips(manifest):
    """Clips verificados de locomoción o reposo de cada caso, con la ruta resuelta del vídeo."""
    manifest = Path(manifest).resolve()
    rows = []
    for case in json.loads(manifest.read_text())['cases']:
        video = (manifest.parent / Path(case['video']).expanduser()).resolve()
        labels = (manifest.parent / Path(case['labels']).expanduser()).resolve()
        table = pd.read_csv(labels)
        table = table[(table.video_id == video.name) & table.reviewed.astype(str).str.lower().isin(REVIEWED) &
                      table.label.isin(CLASSES)]
        for r in table.itertuples():
            rows.append(dict(case=case['name'], clip_id=r.clip_id, video=str(video), video_id=video.name,
                             start=float(r.start_seconds), duration=float(r.duration_seconds), old_label=r.label))
    return pd.DataFrame(rows)


def page(manifest, out, seed=0):
    clips = verified_clips(manifest)
    order = np.random.default_rng(seed).permutation(len(clips))
    clips = clips.iloc[order].reset_index(drop=True)
    clips['code'] = [f'C{seed:02d}{k:03d}' for k in range(len(clips))]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shown = [dict(code=c.code, src=Path(c.video).as_uri(), start=c.start, end=c.start + c.duration) for c in clips.itertuples()]
    html = (PAGE.replace('__CLIPS__', json.dumps(shown)).replace('__SEED__', str(seed))
            .replace('__LOC__', DEFINITION['locomotion']).replace('__REST__', DEFINITION['resting'])
            .replace('__UNK__', DEFINITION['unknown']))
    (out / 'ciego.html').write_text(html, encoding='utf8')
    mapping = clips[['code', 'case', 'clip_id', 'video_id', 'start', 'duration']]
    mapping.to_csv(out / 'ciego_mapa.csv', index=False)
    missing = [v for v in clips.video.unique() if not Path(v).is_file()]
    if missing:
        print('AVISO: vídeos no encontrados:', missing)
    print(f'{len(clips)} clips → {out / "ciego.html"} (mapa aparte: {out / "ciego_mapa.csv"})')
    return out / 'ciego.html'


def kappa(a, b):
    labels = sorted(set(a) | set(b))
    table = pd.crosstab(pd.Categorical(a, labels), pd.Categorical(b, labels), dropna=False).to_numpy(float)
    n = table.sum()
    observed = np.trace(table) / n
    expected = (table.sum(0) * table.sum(1)).sum() / n ** 2
    return float((observed - expected) / (1 - expected)) if expected < 1 else 1.


def apply(manifest, answers, mapping, labels_out, manifest_out, report_out):
    manifest, answers_path = Path(manifest).resolve(), Path(answers)
    answers = pd.read_csv(answers_path)
    mapping = pd.read_csv(mapping)
    if answers.code.duplicated().any() or not set(answers.label) <= {'locomotion', 'resting', 'unknown'}:
        raise ValueError('Respuestas duplicadas o etiquetas no válidas.')
    missing = sorted(set(mapping.code) - set(answers.code))
    if missing:
        raise ValueError(f'Faltan {len(missing)} clips por responder: {missing[:5]}…')
    merged = mapping.merge(answers, on='code', validate='one_to_one')
    old = verified_clips(manifest).set_index('clip_id').old_label
    merged['old_label'] = merged.clip_id.map(old)
    rows = pd.DataFrame(dict(clip_id=merged.clip_id, video_id=merged.video_id, start_seconds=merged.start,
                             duration_seconds=merged.duration, label=merged.label, reviewed='si',
                             review_status='ciego_definicion_fisica_2026-10-27',
                             notes='Reetiquetado ciego; definición física fijada antes; etiqueta anterior: ' + merged.old_label))
    labels_out, manifest_out = Path(labels_out).resolve(), Path(manifest_out).resolve()
    rows.sort_values(['video_id', 'start_seconds']).to_csv(labels_out, index=False)
    spec = json.loads(manifest.read_text())
    for case in spec['cases']:
        for key, value in list(case.items()):
            if key != 'name' and not value.startswith(('~', '/')):  # rutas relativas: relativas al nuevo manifest
                case[key] = os.path.relpath((manifest.parent / value).resolve(), manifest_out.parent)
        case['labels'] = os.path.relpath(labels_out, manifest_out.parent)
    manifest_out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + '\n')
    confusion = pd.crosstab(merged.old_label, merged.label)
    report = dict(definition=DEFINITION, clips=len(merged), new_counts=merged.label.value_counts().to_dict(),
                  agreement=float((merged.old_label == merged.label).mean()),
                  kappa_old_vs_new=kappa(merged.old_label, merged.label),
                  confusion_old_rows_new_columns=confusion.to_dict('index'),
                  changed=merged[merged.old_label != merged.label][['case', 'clip_id', 'old_label', 'label']].to_dict('records'),
                  labels_sha256=sha256(labels_out), answers_sha256=sha256(answers_path),
                  limitation=('Algunos clips se vieron antes con la predicción de CoTracker superpuesta; el reetiquetado es '
                              'ciego al orden y a las etiquetas anteriores, pero no a ese recuerdo.'))
    Path(report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(report_out).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(confusion.to_string())
    print(f"concordancia {report['agreement']:.2f}, kappa {report['kappa_old_vs_new']:.2f}, nuevas: {report['new_counts']}")
    return report


def main():
    parser = argparse.ArgumentParser(description='Reetiquetado ciego con definición física de locomoción')
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('page')
    command.add_argument('--manifest', required=True)
    command.add_argument('--out', default=str(REPO / 'runs' / 'ciego'))
    command.add_argument('--seed', type=int, default=0)
    command = sub.add_parser('apply')
    command.add_argument('--manifest', required=True)
    command.add_argument('--answers', required=True)
    command.add_argument('--mapping', default=str(REPO / 'runs' / 'ciego' / 'ciego_mapa.csv'))
    command.add_argument('--labels-out', default=str(REPO.parent / 'RabbitSense_ARCHIVO' / 'etiquetas_ciegas.csv'))
    command.add_argument('--manifest-out', default=str(REPO / 'benchmarks' / 'pose_ciego.json'))
    command.add_argument('--report-out', default=str(REPO / 'cierre' / 'ciego' / 'concordancia.json'))
    args = parser.parse_args()
    if args.command == 'page':
        page(args.manifest, args.out, args.seed)
    else:
        apply(args.manifest, args.answers, args.mapping, args.labels_out, args.manifest_out, args.report_out)


if __name__ == '__main__':
    main()
