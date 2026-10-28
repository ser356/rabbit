"""Etograma de Cinna y etiquetado continuo de actos (fijado el 2026-10-28, antes de etiquetar).

`page`: página local para etiquetar los vídeos de forma continua: mientras se reproduce, una tecla marca desde ese
instante el acto que empieza (no hay clips fijos de 4 s). `sheets`: hojas de fotogramas cada 0,5 s (tiempo real),
recortados alrededor de la máscara SAM2 con su entorno, para etiquetar por inspección visual; `segments` convierte
esas etiquetas por fotograma en segmentos con el mismo formato que la página. `census`: tiempo, episodios, vídeos y
sesiones por acto, y si hay datos suficientes para evaluar cada acto con validación cruzada por sesión.
"""
import argparse
import json
from pathlib import Path

import cv2
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


SHEET = dict(step=.5, columns=8, rows=8, thumb=200, pad=1., quality=75)


def thumbnail(image, mask, size=SHEET['thumb'], pad=SHEET['pad']):
    """Recorte cuadrado alrededor de la máscara, ampliado `pad` veces su lado por cada lado; sin máscara, imagen entera."""
    if mask is not None and np.count_nonzero(mask) >= .001 * mask.size:
        x, y, w, h = cv2.boundingRect((mask > 0).astype(np.uint8))
        side = int(max(w, h) * (1 + 2 * pad))
        cx, cy = x + w / 2, y + h / 2
        side = min(side, max(image.shape[:2]))
        left = int(np.clip(cx - side / 2, 0, max(0, image.shape[1] - side)))
        top = int(np.clip(cy - side / 2, 0, max(0, image.shape[0] - side)))
        crop, masked = image[top:top + side, left:left + side], True
    else:
        crop, masked = image, False
    scale = size / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)
    tile = np.zeros((size, size, 3), np.uint8)
    top, left = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    tile[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    if not masked:
        cv2.rectangle(tile, (0, 0), (size - 1, size - 1), (0, 0, 255), 3)
    return tile, masked


def sheets(manifest, out, settings=SHEET):
    """Hojas por vídeo: filas de `columns` fotogramas cada `step` s en tiempo real, recortados alrededor de Cinna."""
    from .pointtrack import frame_times
    manifest, out = Path(manifest).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in json.loads(manifest.read_text())['cases']:
        video = (manifest.parent / Path(case['video']).expanduser()).resolve()
        masks = (manifest.parent / Path(case['masks']).expanduser()).resolve() if 'masks' in case else None
        times = frame_times(video)
        targets = np.arange(0., times[-1] + 1e-9, settings['step'])
        wanted = {}
        for t in targets:
            wanted.setdefault(int(np.argmin(np.abs(times - t))), []).append(float(t))
        tiles, capture = [], cv2.VideoCapture(str(video))
        try:
            for frame in range(max(wanted) + 1):
                ok, image = capture.read()
                if not ok:
                    break
                for t in wanted.get(frame, []):
                    mask = cv2.imread(str(masks / 'masks' / f'{frame:06d}.png'), cv2.IMREAD_GRAYSCALE) if masks else None
                    tile, masked = thumbnail(image, mask)
                    cv2.putText(tile, f'{t:.1f}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
                    cv2.putText(tile, f'{t:.1f}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                    tiles.append((t, frame, masked, tile))
        finally:
            capture.release()
        per_sheet = settings['columns'] * settings['rows']
        for s in range(0, len(tiles), per_sheet):
            chunk = tiles[s:s + per_sheet]
            lines = []
            for r in range(0, len(chunk), settings['columns']):
                row = chunk[r:r + settings['columns']]
                label = np.full((settings['thumb'], 70, 3), 20, np.uint8)
                cv2.putText(label, f'{row[0][0]:.0f}s', (6, settings['thumb'] // 2), cv2.FONT_HERSHEY_SIMPLEX, .6,
                            (255, 255, 255), 1)
                images = [x[3] for x in row] + [np.zeros_like(row[0][3])] * (settings['columns'] - len(row))
                lines.append(np.hstack([label] + images))
            name = f"{case['name']}_{s // per_sheet:02d}.jpg"
            cv2.imwrite(str(out / name), np.vstack(lines), [cv2.IMWRITE_JPEG_QUALITY, settings['quality']])
            rows += [dict(case=case['name'], sheet=name, seconds=t, frame=f, masked=m) for t, f, m, _ in chunk]
        print(f"{case['name']}: {len(tiles)} fotogramas", flush=True)
    pd.DataFrame(rows).to_csv(out / 'fotogramas.csv', index=False)
    return len(rows)


MOTION = dict(window=2., stills=4, size=200, gain=15., quality=80, rows=8)  # ganancia fija: no se normaliza por ventana


def motion_map(crops):
    """Media de |diferencia| entre recortes consecutivos (gris), normalizada: brillo = movimiento dentro de la ventana."""
    if len(crops) < 2:
        return np.zeros(crops[0].shape[:2] if crops else (MOTION['size'],) * 2, np.float32)
    gray = [cv2.GaussianBlur(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0) for c in crops]
    return np.mean([np.abs(b - a) for a, b in zip(gray, gray[1:])], axis=0)


def window_box(masks, frames, shape):
    """Caja cuadrada fija para toda la ventana: unión de las máscaras ampliada `pad`; None si no hay máscara."""
    boxes = []
    for f in frames:
        mask = cv2.imread(str(masks / 'masks' / f'{f:06d}.png'), cv2.IMREAD_GRAYSCALE) if masks else None
        if mask is not None and np.count_nonzero(mask) >= .001 * mask.size:
            boxes.append(cv2.boundingRect((mask > 0).astype(np.uint8)))
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    side = min(int(max(x1 - x0, y1 - y0) * (1 + 2 * SHEET['pad'] / 2)), max(shape[:2]))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(np.clip(cx - side / 2, 0, max(0, shape[1] - side)))
    top = int(np.clip(cy - side / 2, 0, max(0, shape[0] - side)))
    return left, top, side


def square(image, box, size):
    if box is None:
        crop = image
    else:
        left, top, side = box
        crop = image[top:top + side, left:left + side]
    scale = size / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)
    tile = np.zeros((size, size, 3), np.uint8)
    y, x = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    tile[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
    return tile


def motion_sheets(manifest, out, settings=MOTION):
    """Hojas de movimiento: por cada ventana de 2 s (tiempo real), 4 fotogramas y un mapa de movimiento calculado con
    todos los fotogramas, sobre una caja fija (sin seguir al conejo, para que su desplazamiento también se vea)."""
    from .pointtrack import frame_times
    manifest, out = Path(manifest).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in json.loads(manifest.read_text())['cases']:
        video = (manifest.parent / Path(case['video']).expanduser()).resolve()
        masks = (manifest.parent / Path(case['masks']).expanduser()).resolve() if 'masks' in case else None
        times = frame_times(video)
        starts = np.arange(0., times[-1] - 1e-9, settings['window'])
        members = {k: np.flatnonzero((times >= s) & (times < s + settings['window'])) for k, s in enumerate(starts)}
        capture = cv2.VideoCapture(str(video))
        images = []
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                scale = 640 / max(image.shape[:2])
                images.append(cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                                         interpolation=cv2.INTER_AREA) if scale < 1 else image)
        finally:
            capture.release()
        lines = []
        for k, start in enumerate(starts):
            frames = [int(f) for f in members[k] if f < len(images)]
            if len(frames) < 2:
                continue
            shape = images[frames[0]].shape
            scale_mask = None
            if masks:
                probe = cv2.imread(str(masks / 'masks' / f'{frames[0]:06d}.png'), cv2.IMREAD_GRAYSCALE)
                scale_mask = None if probe is None else shape[1] / probe.shape[1]
            box = window_box(masks, frames, (int(shape[0] / (scale_mask or 1)), int(shape[1] / (scale_mask or 1)), 3))
            if box is not None and scale_mask:
                box = tuple(int(round(v * scale_mask)) for v in box)
            crops = [square(images[f], box, settings['size']) for f in frames]
            energy = motion_map(crops)
            heat = cv2.applyColorMap(np.clip(energy * settings['gain'], 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            picks = [crops[int(round(j))] for j in np.linspace(0, len(crops) - 1, settings['stills'])]
            label = np.full((settings['size'], 70, 3), 20, np.uint8)
            cv2.putText(label, f'{start:.0f}s', (6, settings['size'] // 2), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            if box is None:
                cv2.putText(label, 'sin', (6, settings['size'] // 2 + 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            lines.append(np.hstack([label] + picks + [heat]))
            rows.append(dict(case=case['name'], start=float(start), end=float(start + settings['window']),
                             frames=len(frames), masked=box is not None,
                             motion_mean=float(energy.mean()), motion_p999=float(np.percentile(energy, 99.9))))
        for s in range(0, len(lines), settings['rows']):
            name = f"{case['name']}_{s // settings['rows']:02d}.jpg"
            cv2.imwrite(str(out / name), np.vstack(lines[s:s + settings['rows']]), [cv2.IMWRITE_JPEG_QUALITY, settings['quality']])
        print(f"{case['name']}: {len(lines)} ventanas", flush=True)
    pd.DataFrame(rows).to_csv(out / 'ventanas.csv', index=False)
    return len(rows)


def window_segments(window_labels, out):
    """Etiquetas por ventana (case, start, act) → segmentos fusionados con el formato de la página."""
    table = pd.read_csv(window_labels).sort_values(['case', 'start'])
    rows = []
    for r in table.itertuples():
        end = float(r.start) + MOTION['window'] if not hasattr(r, 'end') else float(r.end)
        if rows and rows[-1]['video'] == r.case and rows[-1]['act'] == r.act and abs(rows[-1]['end_seconds'] - r.start) < 1e-6:
            rows[-1]['end_seconds'] = end
        else:
            rows.append(dict(video=r.case, start_seconds=float(r.start), end_seconds=end, act=r.act, finished=1))
    pd.DataFrame(rows).to_csv(out, index=False)
    return load(out)


def segments(frame_labels, frames, out, step=SHEET['step']):
    """Etiquetas por fotograma (case, seconds, act) → segmentos [t, t + step) fusionados, formato de la página."""
    labels = pd.read_csv(frame_labels)
    index = pd.read_csv(frames)
    merged = index.merge(labels, on=['case', 'seconds'], how='left', validate='one_to_one')
    if merged.act.isna().any():
        missing = merged[merged.act.isna()][['case', 'seconds']].head().to_dict('records')
        raise ValueError(f'Faltan {int(merged.act.isna().sum())} fotogramas por etiquetar, p. ej. {missing}')
    rows = []
    for case, group in merged.sort_values(['case', 'seconds']).groupby('case', sort=False):
        end = float(group.seconds.max()) + step
        for r in group.itertuples():
            start, stop = float(r.seconds), min(float(r.seconds) + step, end)
            if rows and rows[-1]['video'] == case and rows[-1]['act'] == r.act and abs(rows[-1]['end_seconds'] - start) < 1e-6:
                rows[-1]['end_seconds'] = stop
            else:
                rows.append(dict(video=case, start_seconds=start, end_seconds=stop, act=r.act, finished=1))
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    load(out)
    return table


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


VERIFY = dict(fragments=30, seconds=2., seed=0)
VERIFY_PAGE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Verificación de actos</title>
<style>body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#111;color:#eee}
header{padding:10px 16px;background:#1d1d1d;font-size:13px;line-height:1.45}main{text-align:center;padding:12px}
video{max-width:96vw;max-height:66vh;background:#000}button{font-size:15px;margin:3px;padding:6px 10px}
#status{color:#aaa;margin:6px}</style></head><body>
<header><b>Verificación ciega.</b> Cada fragmento de 2 s se repite en bucle. Pulsa la tecla del acto que domina el
fragmento (si coinciden varios, el primero de la lista). ← → para moverte. Las respuestas se guardan en el navegador.<br>
__LEGEND__</header><main><div id="status"></div><video id="v" muted playsinline></video><div id="keys"></div>
<button id="save">Descargar CSV</button></main><script>
const items = __ITEMS__, acts = __ACTS__, key = 'rabbitsense_verificacion_actos';
let answers = {}; try { answers = JSON.parse(localStorage.getItem(key) || '{}'); } catch (e) {}
let i = Math.max(0, items.findIndex(x => !answers[x.code])); const v = document.getElementById('v');
acts.forEach(a => { const b = document.createElement('button'); b.textContent = a.key + ' · ' + a.name;
  b.onclick = () => answer(a.code); document.getElementById('keys').appendChild(b); });
function show() { const x = items[i]; if (v.dataset.src !== x.src) { v.src = x.src; v.dataset.src = x.src; }
  v.currentTime = x.start; v.play();
  document.getElementById('status').textContent = `Fragmento ${i + 1} de ${items.length} · respondidos ` +
    `${items.filter(y => answers[y.code]).length}` + (answers[x.code] ? ` · tu respuesta: ${answers[x.code]}` : ''); }
v.onloadedmetadata = () => { v.currentTime = items[i].start; v.play(); };
v.ontimeupdate = () => { if (v.currentTime >= items[i].end) v.currentTime = items[i].start; };
function answer(code) { answers[items[i].code] = code; try { localStorage.setItem(key, JSON.stringify(answers)); } catch (e) {}
  if (i < items.length - 1) i++; show(); }
document.getElementById('save').onclick = () => {
  const rows = ['code,act'].concat(items.filter(x => answers[x.code]).map(x => `${x.code},${answers[x.code]}`));
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([rows.join('\\n') + '\\n'], {type: 'text/csv'}));
  a.download = 'verificacion_actos.csv'; a.click(); };
document.addEventListener('keydown', e => { const a = acts.find(x => x.key === e.key); if (a) return answer(a.code);
  if (e.key === 'ArrowLeft') { i = Math.max(0, i - 1); show(); } if (e.key === 'ArrowRight') { i = Math.min(items.length - 1, i + 1); show(); } });
show();
</script></body></html>
"""


def verify_page(manifest, labels, out, settings=VERIFY):
    """Fragmentos de 2 s dentro de un único segmento, repartidos por acto (estratificado) y al azar; sin la etiqueta."""
    table = load(labels)
    table = table[(table.end_seconds - table.start_seconds) >= settings['seconds'] - 1e-6]
    rng = np.random.default_rng(settings['seed'])
    per_act = {act: group for act, group in table.groupby('act')}
    quota = {act: 0 for act in per_act}
    for k in range(settings['fragments']):  # reparto por turnos: actos raros también aparecen
        act = sorted(per_act)[k % len(per_act)]
        quota[act] += 1
    chosen = []
    for act, group in per_act.items():
        weights = (group.end_seconds - group.start_seconds - settings['seconds']).clip(lower=.01).to_numpy()
        for j in rng.choice(len(group), size=quota[act], p=weights / weights.sum()):
            seg = group.iloc[j]
            start = float(rng.uniform(seg.start_seconds, seg.end_seconds - settings['seconds']))
            chosen.append(dict(video=seg.video, start=round(start, 2), end=round(start + settings['seconds'], 2), act=act))
    chosen = [chosen[k] for k in rng.permutation(len(chosen))]
    videos = {v['name']: v['video'] for v in videos_of(manifest)}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for k, item in enumerate(chosen):
        item['code'] = f'V{k:03d}'
    shown = [dict(code=x['code'], src=videos[x['video']].as_uri(), start=x['start'], end=x['end']) for x in chosen]
    legend = ' · '.join(f"<b>{a['key']}</b> {a['name']}: {a['definition']}" for a in ETHOGRAM)
    html = (VERIFY_PAGE.replace('__ITEMS__', json.dumps(shown)).replace('__LEGEND__', legend)
            .replace('__ACTS__', json.dumps([dict(key=a['key'], code=a['code'], name=a['name']) for a in ETHOGRAM])))
    (out / 'verificacion.html').write_text(html, encoding='utf8')
    pd.DataFrame(chosen)[['code', 'video', 'start', 'end', 'act']].to_csv(out / 'verificacion_mapa.csv', index=False)
    print(f'{len(chosen)} fragmentos → {out / "verificacion.html"}')
    return out / 'verificacion.html'


def agreement(answers, mapping, out=None):
    """Acuerdo entre las etiquetas de Claude (mapa) y las de la persona: global, kappa, por acto y matriz."""
    merged = pd.read_csv(mapping).merge(pd.read_csv(answers).rename(columns={'act': 'human'}), on='code', validate='one_to_one')
    per_act = {act: dict(n=int(len(g)), agree=float((g.act == g.human).mean())) for act, g in merged.groupby('act')}
    report = dict(fragments=len(merged), agreement=float((merged.act == merged.human).mean()),
                  kappa=blind_kappa(merged.act, merged.human), per_claude_act=per_act,
                  confusion_claude_rows_human_columns=pd.crosstab(merged.act, merged.human).to_dict('index'),
                  answers_sha256=sha256(answers))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(pd.crosstab(merged.act, merged.human).to_string())
    print(f"acuerdo {report['agreement']:.2f}, kappa {report['kappa']:.2f} sobre {len(merged)} fragmentos")
    return report


def blind_kappa(a, b):
    from .blind import kappa
    return kappa(list(a), list(b))


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
    command = sub.add_parser('sheets')
    command.add_argument('--manifest', required=True)
    command.add_argument('--out', default=str(REPO / 'runs' / 'actos_hojas'))
    command = sub.add_parser('motion-sheets')
    command.add_argument('--manifest', required=True)
    command.add_argument('--out', default=str(REPO / 'runs' / 'actos_movimiento'))
    command = sub.add_parser('segments')
    command.add_argument('--labels', required=True, help='CSV case,seconds,act (una fila por fotograma)')
    command.add_argument('--frames', required=True, help='fotogramas.csv de sheets')
    command.add_argument('--out', required=True)
    command = sub.add_parser('verify-page')
    command.add_argument('--manifest', required=True)
    command.add_argument('--labels', required=True)
    command.add_argument('--out', default=str(REPO / 'runs' / 'actos_verificacion'))
    command = sub.add_parser('agreement')
    command.add_argument('--answers', required=True)
    command.add_argument('--mapping', default=str(REPO / 'runs' / 'actos_verificacion' / 'verificacion_mapa.csv'))
    command.add_argument('--out', default=str(REPO / 'cierre' / 'actos' / 'acuerdo.json'))
    command = sub.add_parser('census')
    command.add_argument('--labels', required=True)
    command.add_argument('--out', default=str(REPO / 'cierre' / 'actos' / 'censo.json'))
    args = parser.parse_args()
    if args.command == 'page':
        page(args.manifest, args.out)
    elif args.command == 'motion-sheets':
        print('ventanas:', motion_sheets(args.manifest, args.out))
    elif args.command == 'sheets':
        print('fotogramas:', sheets(args.manifest, args.out))
    elif args.command == 'verify-page':
        verify_page(args.manifest, args.labels, args.out)
    elif args.command == 'agreement':
        agreement(args.answers, args.mapping, args.out)
    elif args.command == 'segments':
        segments(args.labels, args.frames, args.out)
    else:
        census(args.labels, args.out)


if __name__ == '__main__':
    main()
