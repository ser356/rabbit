"""Etograma de Cinna y etiquetado continuo de actos (fijado el 2026-10-28, antes de etiquetar).

`page`: página local para etiquetar los vídeos de forma continua: mientras se reproduce, una tecla marca desde ese
instante el acto que empieza (no hay clips fijos de 4 s). `census`: tiempo, episodios, vídeos y sesiones por acto,
y si hay datos suficientes para evaluar cada acto con validación cruzada por sesión.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .motion import sha256

REPO = Path(__file__).resolve().parents[2]
# Un solo acto a la vez. Si coinciden varios, manda el primero de esta lista (orden de prioridad).
ETHOGRAM = [
    dict(key='8', code='binky', name='Binky', definition='Salto con giro del cuerpo o pataleo en el aire.'),
    dict(key='7', code='rearing', name='De pie', definition='Erguida sobre las patas traseras, patas delanteras sin apoyar.'),
    dict(key='1', code='locomotion', name='Desplazarse',
         definition='Cambia de sitio sobre el suelo con el cuerpo entero (al menos medio cuerpo), aunque sea despacio.'),
    dict(key='2', code='eating', name='Comer', definition='Mastica o arranca comida (heno, verdura, pienso) con el hocico en ella.'),
    dict(key='3', code='grooming', name='Acicalarse', definition='Se lame o se frota cara, orejas o cuerpo con la lengua o las patas.'),
    dict(key='4', code='exploring', name='Explorar u olfatear',
         definition='Hocico al suelo u objetos, cabeza en movimiento; puede avanzar menos de medio cuerpo.'),
    dict(key='6', code='lying', name='Tumbada o flop',
         definition='Tumbada de lado o estirada con las patas extendidas; incluye dejarse caer de lado (flop).'),
    dict(key='5', code='loaf', name='Agazapada', definition='Quieta con las patas recogidas bajo el cuerpo, en bola.'),
    dict(key='0', code='not_visible', name='No visible u otro',
         definition='No se ve bien, la tocan o la sujetan, o ninguna de las anteriores.'),
]
CODES = [a['code'] for a in ETHOGRAM]
WINDOW = 2.  # ventanas de análisis previstas
# Mínimos fijados antes de etiquetar para evaluar un acto con validación cruzada por sesión.
EVALUABLE = dict(min_disjoint_windows=20, min_sessions=3, min_bouts=5)

PAGE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Etiquetado continuo de actos</title>
<style>
body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#111;color:#eee}
header{padding:10px 16px;background:#1d1d1d;font-size:13px;line-height:1.45}
main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:12px;padding:12px}
video{width:100%;max-height:64vh;background:#000}
#bar{position:relative;height:26px;background:#2a2a2a;margin-top:6px;cursor:pointer}
#bar div{position:absolute;top:0;height:100%}
#cursor{width:2px;background:#fff}
aside{font-size:13px}
select,button{font-size:14px;margin:2px 0}
table{border-collapse:collapse;width:100%}
td{padding:2px 4px;border-bottom:1px solid #333}
.k{display:inline-block;min-width:16px;padding:0 4px;border-radius:3px;color:#000;font-weight:600;text-align:center}
</style></head><body>
<header><b>Etiquetado continuo.</b> Mientras se reproduce, pulsa la tecla del acto que <b>empieza</b> en ese instante;
dura hasta la siguiente tecla. Si coinciden varios, usa el primero de la lista. Espacio: pausa · ← →: ±1 s ·
, .: velocidad · Retroceso: borra la última marca · Las marcas se guardan en este navegador.<br>__LEGEND__</header>
<main><section>
<select id="videos"></select> <span id="status"></span>
<video id="v" controls muted playsinline></video>
<div id="bar"><div id="cursor"></div></div>
</section><aside>
<button id="save">Descargar CSV de todos los vídeos</button>
<button id="done">Marcar vídeo como terminado</button>
<table id="marks"></table></aside></main>
<script>
const videos = __VIDEOS__, acts = __ACTS__, key = 'rabbitsense_actos_v1';
const colour = {binky:'#f0f',rearing:'#ff0',locomotion:'#f80',eating:'#0c4',grooming:'#0cf',exploring:'#9f6',
  lying:'#69f',loaf:'#36c',not_visible:'#777'};
let state = {}; try { state = JSON.parse(localStorage.getItem(key) || '{}'); } catch (e) {}
const v = document.getElementById('v'), sel = document.getElementById('videos');
videos.forEach((x, i) => { const o = document.createElement('option'); o.value = i; o.textContent = x.name; sel.appendChild(o); });
let current = 0;
function marks() { return (state[videos[current].name] = state[videos[current].name] || {marks: [], done: false}); }
function persist() { try { localStorage.setItem(key, JSON.stringify(state)); } catch (e) {} draw(); }
function segments(name) {
  const m = (state[name] || {marks: []}).marks.slice().sort((a, b) => a.t - b.t);
  const end = (state[name] || {}).end || videos.find(x => x.name === name).end;
  return m.map((x, i) => ({start: x.t, end: i + 1 < m.length ? m[i + 1].t : end, act: x.act}));
}
function draw() {
  const name = videos[current].name, bar = document.getElementById('bar'), end = v.duration || videos[current].end;
  bar.querySelectorAll('.seg').forEach(e => e.remove());
  segments(name).forEach(s => { const d = document.createElement('div'); d.className = 'seg';
    d.style.left = (100 * s.start / end) + '%'; d.style.width = (100 * (s.end - s.start) / end) + '%';
    d.style.background = colour[s.act]; d.title = s.act; bar.insertBefore(d, bar.firstChild); });
  document.getElementById('marks').innerHTML = segments(name).map(s =>
    `<tr><td>${s.start.toFixed(2)}–${s.end.toFixed(2)} s</td><td><span class="k" style="background:${colour[s.act]}">` +
    `${acts.find(a => a.code === s.act).name}</span></td></tr>`).join('');
  const done = Object.values(state).filter(x => x.done).length;
  document.getElementById('status').textContent = `${marks().done ? 'terminado' : 'en curso'} · ${done}/${videos.length} vídeos terminados`;
}
function load(i) { current = i; sel.value = i; v.src = videos[i].src; v.playbackRate = 1; draw(); }
sel.onchange = () => load(+sel.value);
v.ontimeupdate = () => { const end = v.duration || videos[current].end;
  document.getElementById('cursor').style.left = (100 * v.currentTime / end) + '%'; };
v.onloadedmetadata = () => { marks().end = v.duration; persist(); };
document.getElementById('bar').onclick = e => { const r = e.currentTarget.getBoundingClientRect();
  v.currentTime = (e.clientX - r.left) / r.width * (v.duration || videos[current].end); };
document.getElementById('done').onclick = () => { marks().done = !marks().done; persist(); };
document.getElementById('save').onclick = () => {
  const rows = ['video,start_seconds,end_seconds,act,finished'];
  videos.forEach(x => segments(x.name).forEach(s =>
    rows.push(`"${x.name}",${s.start.toFixed(3)},${s.end.toFixed(3)},${s.act},${(state[x.name] || {}).done ? 1 : 0}`)));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([rows.join('\\n') + '\\n'], {type: 'text/csv'}));
  a.download = 'actos_cinna.csv'; a.click();
};
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  const act = acts.find(a => a.key === e.key);
  if (act) { const m = marks().marks.filter(x => Math.abs(x.t - v.currentTime) > 1e-3);
    m.push({t: +v.currentTime.toFixed(3), act: act.code}); marks().marks = m; persist(); return; }
  if (e.key === ' ') { e.preventDefault(); v.paused ? v.play() : v.pause(); }
  else if (e.key === 'ArrowLeft') v.currentTime = Math.max(0, v.currentTime - 1);
  else if (e.key === 'ArrowRight') v.currentTime += 1;
  else if (e.key === ',') v.playbackRate = Math.max(.25, v.playbackRate / 2);
  else if (e.key === '.') v.playbackRate = Math.min(2, v.playbackRate * 2);
  else if (e.key === 'Backspace') { e.preventDefault(); const m = marks().marks.sort((a, b) => a.t - b.t); m.pop(); persist(); }
});
load(0);
</script></body></html>
"""


def session_of(name):
    return 'whatsapp' if name.startswith('WhatsApp') else 'mano_IMG_519x' if name.startswith('IMG_519') else name


def videos_of(manifest):
    manifest = Path(manifest).resolve()
    rows = []
    for case in json.loads(manifest.read_text())['cases']:
        video = (manifest.parent / Path(case['video']).expanduser()).resolve()
        rows.append(dict(name=case['name'], video=video))
    return rows


def page(manifest, out, durations=None):
    """Página de etiquetado; `durations` (nombre → s) solo hace falta si el navegador no da la duración."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shown = [dict(name=v['name'], src=v['video'].as_uri(), end=float((durations or {}).get(v['name'], 0.)))
             for v in videos_of(manifest)]
    legend = ' · '.join(f"<b>{a['key']}</b> {a['name']}: {a['definition']}" for a in ETHOGRAM)
    html = (PAGE.replace('__VIDEOS__', json.dumps(shown)).replace('__LEGEND__', legend)
            .replace('__ACTS__', json.dumps([dict(key=a['key'], code=a['code'], name=a['name']) for a in ETHOGRAM])))
    (out / 'actos.html').write_text(html, encoding='utf8')
    print(f'{len(shown)} vídeos → {out / "actos.html"}')
    return out / 'actos.html'


def load(path):
    table = pd.read_csv(path)
    unknown = set(table.act) - set(CODES)
    if unknown:
        raise ValueError(f'Actos desconocidos: {sorted(unknown)}')
    if (table.end_seconds <= table.start_seconds).any():
        raise ValueError('Hay segmentos de duración nula o negativa.')
    for name, group in table.groupby('video'):
        group = group.sort_values('start_seconds')
        if (group.start_seconds.to_numpy()[1:] < group.end_seconds.to_numpy()[:-1] - 1e-6).any():
            raise ValueError(f'Segmentos solapados en {name}.')
    return table


def census(path, out=None):
    """Por acto: segundos, episodios, vídeos, sesiones y ventanas disjuntas de 2 s; evaluable o no (EVALUABLE)."""
    table = load(path)
    table = table.assign(seconds=table.end_seconds - table.start_seconds, session=table.video.map(session_of),
                         windows=np.floor((table.end_seconds - table.start_seconds) / WINDOW).astype(int))
    rows = []
    for act in CODES:
        part = table[table.act == act]
        row = dict(act=act, seconds=float(part.seconds.sum()), bouts=int(len(part)), videos=int(part.video.nunique()),
                   sessions=int(part.session.nunique()), disjoint_windows=int(part.windows.sum()))
        row['evaluable'] = bool(act != 'not_visible' and row['disjoint_windows'] >= EVALUABLE['min_disjoint_windows'] and
                                row['sessions'] >= EVALUABLE['min_sessions'] and row['bouts'] >= EVALUABLE['min_bouts'])
        rows.append(row)
    result = pd.DataFrame(rows)
    report = dict(acts=rows, criteria=EVALUABLE, labelled_seconds=float(table.seconds.sum()),
                  finished_videos=int(table.groupby('video').finished.max().sum()) if 'finished' in table else None,
                  labels_sha256=sha256(path),
                  note=('Una ventana de 2 s por cada 2 s completos de un mismo episodio; ventanas de un mismo vídeo no son '
                        'independientes: la unidad de evaluación es la sesión.'))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(result.to_string(index=False))
    print(f"evaluables: {', '.join(result[result.evaluable].act) or 'ninguno'}")
    return report


def main():
    parser = argparse.ArgumentParser(description='Etograma de Cinna: etiquetado continuo y recuento de actos')
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('page')
    command.add_argument('--manifest', required=True)
    command.add_argument('--out', default=str(REPO / 'runs' / 'actos'))
    command = sub.add_parser('census')
    command.add_argument('--labels', required=True)
    command.add_argument('--out', default=str(REPO / 'cierre' / 'actos' / 'censo.json'))
    args = parser.parse_args()
    if args.command == 'page':
        page(args.manifest, args.out)
    else:
        census(args.labels, args.out)


if __name__ == '__main__':
    main()
