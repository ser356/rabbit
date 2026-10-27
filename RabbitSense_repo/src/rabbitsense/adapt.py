"""Adaptación del modelo temporal a Cinna: línea experimental nueva (2026-10-27).

No toca los veredictos `not_resolved` anteriores ni sus modelos congelados. Todo se evalúa con validación cruzada
agrupada (por vídeo y, más estricta, por sesión); nada del grupo evaluado entra en el entrenamiento, ni siquiera la
estandarización o la elección de la regularización.

1. `audit`: ejemplos útiles e independientes por vídeo y sesión; ventanas solapadas aparte.
2. `cache`: fotogramas cada 0,25 s (lado largo 448) y caja de la máscara SAM2 de cada uno.
3. `embed`: X-CLIP congelado sobre ventanas de 8 fotogramas de la imagen completa y de un recorte del conejo con
   entorno (unión de cajas de la ventana, ampliada).
4. `probe`: regresiones logísticas sobre esas características (imagen, recorte, ambas, ambas + YouTube) y modelos
   YouTube congelados como referencia; mismos criterios y métricas que el cierre.
5. `finetune`: ajuste parcial de X-CLIP (última capa visual, proyección y MIT) con las dos vistas, por pliegue.
6. `diagnose`: vídeos con predicciones superpuestas y clasificación automática de errores.
"""
import argparse
import json
import platform
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .closure import CRITERIA, judge, tally
from .evaluate import clip_metrics
from .motion import sha256
from .webclips import BAND, FRAMES, LEAD, MODEL_ID, SIZE, SPACING, fit, letterbox, metadata, predict, window_ends

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'cierre' / 'adapt'
REVIEWED = {'si', 'sí', 'yes', 'true'}
CLASSES = ('locomotion', 'resting')
SPAN = (FRAMES - 1) * SPACING  # 1,75 s entre el primer y el último fotograma de una ventana
CACHE_SIDE, JPEG_QUALITY, MIN_MASK = 448, 90, .001
CROP_PAD, CROP_MIN = .5, 48  # el recorte añade medio tamaño de caja por lado: conejo y suelo alrededor
L2_GRID = (1., 10., 100.)  # elegido dentro de cada pliegue con validación interna agrupada
CINNA_WEIGHT = 1.  # con YouTube: el peso total de Cinna iguala al de YouTube
IMAGENET_MEAN, IMAGENET_STD = [.485, .456, .406], [.229, .224, .225]  # si el modelo no trae procesador
# Fijado antes de ejecutar; no se ajusta mirando resultados de los pliegues.
FINETUNE = dict(epochs=4, batch=8, lr_encoder=2e-5, lr_head=1e-3, weight_decay=.05, seed=0,
                trainable='vision_model.encoder.layers[-1], vision_model.post_layernorm, visual_projection, mit')
# Orden de clasificación de errores: se asigna la primera causa que se cumple.
ERROR_RULES = dict(tracking=.5, camera=.02, small=.01, short=1.)


def session_of(name):
    """Agrupación conservadora: vídeos que pueden compartir sesión, sala y luz cuentan como uno."""
    if name.startswith('WhatsApp'):
        return 'whatsapp_2026-10-26'
    if name.startswith('IMG_519'):
        return 'mano_IMG_5190-5193'
    return name


def load_cases(manifest):
    manifest = Path(manifest).resolve()
    cases = []
    for case in json.loads(manifest.read_text())['cases']:
        def path(key):
            return (manifest.parent / Path(case[key]).expanduser()).resolve() if key in case else None
        video, labels = path('video'), path('labels')
        table = pd.read_csv(labels)
        table = table[(table.video_id == video.name) & table.reviewed.astype(str).str.lower().isin(REVIEWED) &
                      table.label.isin(CLASSES)]
        clips = pd.DataFrame(dict(clip_id=table.clip_id.to_numpy(), start=table.start_seconds.to_numpy(float),
                                  end=(table.start_seconds + table.duration_seconds).to_numpy(float),
                                  label=table.label.to_numpy()))
        cases.append(dict(name=case['name'], video=video, labels=labels, masks=path('masks'), clips=clips,
                          session=session_of(case['name'])))
    return cases


def windows_inside(start, end, step):
    return max(0, int(np.floor((end - start - SPAN) / step + 1e-9)) + 1)


def wilson(k, n, z=1.96):
    if not n:
        return None, None
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def clips_needed(observed=.9, bound=.8):
    """Clips necesarios para que, acertando la fracción `observed`, el límite inferior de Wilson supere `bound`."""
    return next(n for n in range(5, 2000) if wilson(round(observed * n), n)[0] >= bound)


def video_info(video):
    info = dict(exists=video.is_file())
    if not info['exists']:
        return info
    fps, count = metadata(video)
    capture = cv2.VideoCapture(str(video))
    info.update(fps=fps, frames=count, seconds=count / fps, width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    try:
        tags = json.loads(subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(video)],
                                         capture_output=True, text=True, timeout=30).stdout)['format'].get('tags', {})
        info['creation_time'] = tags.get('creation_time') or tags.get('com.apple.quicktime.creationdate')
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        info['creation_time'] = None
    return info


def audit(manifest, features=None, cache_dir=None, out=OUT):
    rows = []
    for case in load_cases(manifest):
        row = dict(case=case['name'], session=case['session'], **{f'video_{k}': v for k, v in video_info(case['video']).items()})
        for label in CLASSES:
            clips = case['clips'][case['clips'].label == label]
            row[f'{label}_clips'] = len(clips)
            row[f'{label}_seconds'] = round(float((clips.end - clips.start).sum()), 2)
            row[f'{label}_windows_overlapping'] = sum(windows_inside(c.start, c.end, .5) for c in clips.itertuples())
            row[f'{label}_windows_disjoint'] = sum(windows_inside(c.start, c.end, SPAN + SPACING) for c in clips.itertuples())
        if features and (Path(features) / f"{case['video'].stem}.npz").is_file():
            seconds = np.load(Path(features) / f"{case['video'].stem}.npz")['seconds']
            row['feature_windows_in_labeled_clips'] = int(sum(np.isfinite(window_label(seconds, case['clips']))))
        if cache_dir and (Path(cache_dir) / f"{case['name']}.npz").is_file():
            cache = load_cache(Path(cache_dir) / f"{case['name']}.npz")
            inside = labeled_keys(cache['keys'], case['clips'])
            row['mask_coverage_labeled'] = float(np.isfinite(cache['box'][inside, 0]).mean()) if inside.any() else None
            row['mask_area_median_labeled'] = float(np.nanmedian(cache['area'][inside])) if inside.any() else None
        rows.append(row)
    table = pd.DataFrame(rows)
    loc, rest = table.locomotion_clips.sum(), table.resting_clips.sum()
    by_session = table.groupby('session')[['locomotion_clips', 'resting_clips']].sum()
    report = dict(
        cases=rows,
        totals=dict(videos_with_labels=int(((table.locomotion_clips + table.resting_clips) > 0).sum()),
                    videos_with_locomotion=int((table.locomotion_clips > 0).sum()),
                    videos_with_resting=int((table.resting_clips > 0).sum()),
                    videos_with_both=int(((table.locomotion_clips > 0) & (table.resting_clips > 0)).sum()),
                    sessions=int(len(by_session)), sessions_with_locomotion=int((by_session.locomotion_clips > 0).sum()),
                    sessions_with_resting=int((by_session.resting_clips > 0).sum()),
                    locomotion_clips=int(loc), resting_clips=int(rest),
                    locomotion_seconds=float(table.locomotion_seconds.sum()), resting_seconds=float(table.resting_seconds.sum()),
                    overlapping_windows=int(table.locomotion_windows_overlapping.sum() + table.resting_windows_overlapping.sum()),
                    disjoint_windows=int(table.locomotion_windows_disjoint.sum() + table.resting_windows_disjoint.sum()),
                    largest_video_share=float(((table.locomotion_clips + table.resting_clips) / (loc + rest)).max())),
        sessions=by_session.astype(int).to_dict('index'),
        precision=dict(sensitivity_at_0_8=wilson(round(.8 * loc), loc), specificity_at_0_8=wilson(round(.8 * rest), rest),
                       clips_needed_for_lower_bound_0_8_if_observed_0_9=clips_needed(.9),
                       clips_needed_for_lower_bound_0_8_if_observed_0_95=clips_needed(.95)),
        confound=dict(single_class_videos=int(((table.locomotion_clips > 0) ^ (table.resting_clips > 0)).sum()),
                      single_class_sessions=int(((by_session.locomotion_clips > 0) ^ (by_session.resting_clips > 0)).sum()),
                      note=('En los vídeos de una sola clase, la escena predice la etiqueta: un modelo puede acertar '
                            'reconociendo el vídeo, no el desplazamiento. La AUC intra-vídeo (vídeos con ambas clases) '
                            'lo controla.')),
        independence=('Unidad independiente: la sesión (agrupación conservadora en session_of); el vídeo es la unidad de '
                      'validación pedida. Los clips de 4 s de un mismo vídeo comparten escena y luz: no son '
                      'independientes. Las ventanas cada 0,5 s abarcan 1,75 s y se solapan; las disjuntas son el '
                      'máximo de ventanas sin fotogramas compartidos.'),
        manifest_sha256=sha256(manifest))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'auditoria.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float))
    columns = ['case', 'session', 'locomotion_clips', 'resting_clips', 'locomotion_windows_overlapping',
               'resting_windows_overlapping', 'locomotion_windows_disjoint', 'resting_windows_disjoint']
    columns += [c for c in ('feature_windows_in_labeled_clips', 'mask_coverage_labeled', 'video_creation_time') if c in table]
    print(table[columns].to_string(index=False))
    print(json.dumps(report['totals'], indent=2), json.dumps(report['precision'], indent=2), sep='\n')
    return report


def window_label(seconds, clips):
    """1 locomoción, 0 reposo, NaN si la ventana que termina en t no cae entera dentro de un clip etiquetado."""
    labels = np.full(len(seconds), np.nan)
    for c in clips.itertuples():
        start = np.asarray(seconds) - LEAD
        labels[(start >= c.start - 1e-6) & (start + SPAN <= c.end + 1e-6)] = 1. if c.label == 'locomotion' else 0.
    return labels


def labeled_keys(keys, clips):
    t = np.asarray(keys) * SPACING
    return np.array([any(c.start - 1e-6 <= x <= c.end + 1e-6 for c in clips.itertuples()) for x in t], bool)


# --- caché de fotogramas ---------------------------------------------------------------------------------------

def mask_box(mask):
    if mask is None or np.count_nonzero(mask) < MIN_MASK * mask.size:
        return [np.nan] * 4, 0., [np.nan] * 2
    x, y, w, h = cv2.boundingRect((mask > 0).astype(np.uint8))
    m = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    return [x, y, w, h], float(np.count_nonzero(mask) / mask.size), [m['m10'] / m['m00'], m['m01'] / m['m00']]


def cache_case(case, out):
    video = case['video']
    fps, count = metadata(video)
    masks = case['masks'] if case['masks'] and (case['masks'] / 'masks').is_dir() else None
    if masks is not None:
        summary = json.loads((masks / 'summary.json').read_text())
        if summary['inputs']['video']['sha256'] != sha256(video):
            raise ValueError(f"Las máscaras de {masks} no corresponden a {video}.")
    wanted = {}
    for k in range(int(count / fps / SPACING) + 1):
        frame = int(round(k * SPACING * fps))
        if frame < count:
            wanted.setdefault(frame, []).append(k)
    keys, jpegs, boxes, areas, centres, scale = [], [], [], [], [], None
    capture = cv2.VideoCapture(str(video))
    try:
        for frame in range(max(wanted) + 1):
            ok, image = capture.read()
            if not ok:
                break
            if frame not in wanted:
                continue
            scale = CACHE_SIDE / max(image.shape[:2])
            small = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            mask = cv2.imread(str(masks / 'masks' / f'{frame:06d}.png'), cv2.IMREAD_GRAYSCALE) if masks else None
            box, area, centre = mask_box(mask)
            encoded = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
            for k in wanted[frame]:
                keys.append(k)
                jpegs.append(encoded)
                boxes.append(np.array(box) * scale)
                areas.append(area)
                centres.append(np.array(centre) * scale)
    finally:
        capture.release()
    lengths = np.array([len(j) for j in jpegs])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{case['name']}.npz", keys=np.array(keys), box=np.array(boxes, float), area=np.array(areas),
             centre=np.array(centres, float), offsets=np.r_[0, np.cumsum(lengths)],
             jpeg=np.frombuffer(b''.join(jpegs), np.uint8), fps=fps, count=count, has_masks=masks is not None,
             video_sha256=sha256(video))
    return len(keys)


def load_cache(path):
    data = dict(np.load(path))
    data['index'] = {int(k): i for i, k in enumerate(data['keys'])}
    return data


def frame_of(cache, k):
    i = cache['index'][k]
    return cv2.imdecode(cache['jpeg'][cache['offsets'][i]:cache['offsets'][i + 1]], cv2.IMREAD_COLOR), i


def crop_region(boxes, shape, jitter=None):
    """Unión de las cajas válidas de la ventana, ampliada y cuadrada; None si no hay ninguna."""
    valid = boxes[np.isfinite(boxes[:, 0])]
    if not len(valid):
        return None
    x0, y0 = valid[:, 0].min(), valid[:, 1].min()
    x1, y1 = (valid[:, 0] + valid[:, 2]).max(), (valid[:, 1] + valid[:, 3]).max()
    side = max(x1 - x0, y1 - y0, CROP_MIN) * (1 + 2 * CROP_PAD)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if jitter is not None:
        side *= jitter[0]
        cx, cy = cx + jitter[1] * side, cy + jitter[2] * side
    side = min(side, max(shape[:2]))
    left = int(np.clip(cx - side / 2, 0, max(0, shape[1] - side)))
    top = int(np.clip(cy - side / 2, 0, max(0, shape[0] - side)))
    return left, top, int(min(shape[1], left + side)), int(min(shape[0], top + side))


def window_images(cache, keys, rng=None):
    """Ventana de 8 fotogramas: imagen completa y recorte del conejo con entorno, ambas 224 RGB."""
    frames, rows = zip(*(frame_of(cache, k) for k in keys))
    jitter = None if rng is None else (rng.uniform(.85, 1.15), rng.uniform(-.08, .08), rng.uniform(-.08, .08))
    region = crop_region(cache['box'][list(rows)], frames[0].shape, jitter)
    full = np.stack([letterbox(f, SIZE) for f in frames])
    crop = full if region is None else np.stack([letterbox(f[region[1]:region[3], region[0]:region[2]], SIZE)
                                                 for f in frames])
    if rng is not None:
        if rng.random() < .5:
            full, crop = full[:, :, ::-1], crop[:, :, ::-1]
        gain, bias = rng.uniform(.8, 1.2), rng.uniform(-20, 20)
        full, crop = (np.clip(a.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8) for a in (full, crop))
    return full, crop, float(region is not None)


def grid(cache):
    """Instantes de la rejilla de 0,5 s con todos sus fotogramas en caché, y sus claves."""
    ends = [t for t in window_ends(int(cache['count']), float(cache['fps']))
            if (first := int(round((t - LEAD) / SPACING))) >= 0 and all(k in cache['index'] for k in range(first, first + FRAMES))]
    return np.array(ends), [list(range(int(round((t - LEAD) / SPACING)), int(round((t - LEAD) / SPACING)) + FRAMES)) for t in ends]


# --- codificador ----------------------------------------------------------------------------------------------

class Encoder:
    """X-CLIP con la normalización de su procesador aplicada a mano, para poder entrenar parte de él.

    Las ventanas ya son de 224×224, así que el redimensionado y el recorte central del procesador no cambian nada.
    """

    def __init__(self, device='cpu', model_id=MODEL_ID, trainable=False):
        import torch
        import transformers
        self.torch, self.device = torch, device
        self.model = transformers.XCLIPModel.from_pretrained(model_id).to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        if trainable:
            for module in (self.model.vision_model.encoder.layers[-1], self.model.vision_model.post_layernorm,
                           self.model.visual_projection, self.model.mit):
                for p in module.parameters():
                    p.requires_grad_(True)
        self.model.train(trainable)
        try:
            processor = transformers.XCLIPProcessor.from_pretrained(model_id).image_processor
            mean, std = list(processor.image_mean), list(processor.image_std)
        except (OSError, ValueError, TypeError):
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 1, 3, 1, 1)
        self.info = dict(model_id=model_id, revision=getattr(self.model.config, '_commit_hash', None),
                         torch=torch.__version__, transformers=transformers.__version__, device=device,
                         image_mean=mean, image_std=std)

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def __call__(self, videos):
        """videos: uint8 (B, 8, 224, 224, 3) → (B, D)."""
        x = self.torch.as_tensor(np.ascontiguousarray(videos), device=self.device).permute(0, 1, 4, 2, 3).float() / 255.
        output = self.model.get_video_features(pixel_values=(x - self.mean) / self.std)
        return output if isinstance(output, self.torch.Tensor) else output.pooler_output


def embed(manifest, cache_dir, out, encoder, batch=16, reference=None):
    """`reference`: características de webclips para comprobar que la vista completa coincide (coseno)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for case in load_cases(manifest):
        cache = load_cache(Path(cache_dir) / f"{case['name']}.npz")
        ends, keys = grid(cache)
        full, crop, valid = [], [], []
        for start in range(0, len(ends), batch):
            windows = [window_images(cache, k) for k in keys[start:start + batch]]
            with encoder.torch.no_grad():
                full.append(encoder(np.stack([w[0] for w in windows])).float().cpu().numpy())
                crop.append(encoder(np.stack([w[1] for w in windows])).float().cpu().numpy())
            valid += [w[2] for w in windows]
        np.savez(out / f"{case['name']}.npz", seconds=ends, full=np.concatenate(full), crop=np.concatenate(crop),
                 crop_valid=np.array(valid))
        check = ''
        if reference and (Path(reference) / f"{case['video'].stem}.npz").is_file():
            old = np.load(Path(reference) / f"{case['video'].stem}.npz")
            index = {round(float(s), 3): i for i, s in enumerate(old['seconds'])}
            pairs = [(i, index[round(float(s), 3)]) for i, s in enumerate(ends) if round(float(s), 3) in index]
            if pairs:
                a = np.concatenate(full)[[i for i, _ in pairs]]
                b = old['features'][[j for _, j in pairs]]
                cosine = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
                check = f'  coseno con webclips {np.median(cosine):.3f}'
        print('CARACTERÍSTICAS', case['name'], len(ends), f'recorte válido {np.mean(valid):.2f}{check}', flush=True)
    (out / 'embedder.json').write_text(json.dumps(encoder.info, indent=2))


# --- validación cruzada agrupada ------------------------------------------------------------------------------

def fit_weighted(x, y, sample_weight, l2):
    """Regresión logística L2 por Newton con pesos por muestra, sobre características estandarizadas."""
    mean, scale = x.mean(axis=0), x.std(axis=0) + 1e-6
    z = np.c_[np.ones(len(x)), (x - mean) / scale]
    w, penalty = np.zeros(z.shape[1]), np.r_[0., np.full(z.shape[1] - 1, l2)]
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))
        gradient = z.T @ (sample_weight * (p - y)) + penalty * w
        hessian = (z * (sample_weight * p * (1 - p))[:, None]).T @ z + np.diag(penalty + 1e-9)
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return dict(mean=mean.tolist(), scale=scale.tolist(), weights=w.tolist())


def balanced(y):
    return np.where(y == 1, .5 / max(y.mean(), 1e-9), .5 / max(1 - y.mean(), 1e-9))


def auc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    positive, negative = p[y == 1], p[y == 0]
    if not len(positive) or not len(negative):
        return None
    return float((positive[:, None] > negative[None]).mean() + .5 * (positive[:, None] == negative[None]).mean())


def choose_l2(x, y, groups):
    """L2 por validación interna agrupada sobre los datos de entrenamiento del pliegue (AUC de ventanas)."""
    names = sorted(set(groups))
    if len(names) < 3:
        return L2_GRID[1]
    scores = {}
    for l2 in L2_GRID:
        held = np.full(len(y), np.nan)
        for g in names:
            test = groups == g
            if len(set(y[~test])) == 2:
                held[test] = predict(fit(x[~test], y[~test], l2), x[test])
        ok = np.isfinite(held)
        scores[l2] = auc(held[ok], y[ok]) or 0.
    return max(L2_GRID, key=lambda l2: (scores[l2], -abs(np.log10(l2) - 1)))


def feature_set(data, name):
    if name == 'full':
        return data['full']
    if name == 'crop':
        return data['crop']
    return np.c_[data['full'], data['crop'], data['crop_valid']]


def predictions_to_rows(ends, p, fps, count):
    """Mismo convenio que webclips.classify: la probabilidad de la ventana que termina en t decide (t − 0,5, t]."""
    value = dict(zip(np.round(ends, 3), p))
    rows, grid_all = [], window_ends(count, fps)
    for index, t in enumerate(grid_all):
        row = dict(seconds=t, interval_seconds=t - grid_all[index - 1] if index else 0., movement='unknown',
                   p_locomotion=None, evidence_quality=0.)
        q = value.get(round(t, 3))
        if index and q is not None:
            row.update(p_locomotion=float(q), evidence_quality=abs(2 * float(q) - 1),
                       movement='locomotion_candidate' if q >= BAND[1] else 'low_motion_candidate' if q <= BAND[0] else 'unknown')
        rows.append(row)
    return rows


def score(cases, predictions, caches):
    """Métricas del cierre sobre predicciones fuera de pliegue: detección por clip, AUC y cobertura."""
    per_case, windows_p, windows_y, clip_p, clip_y, known, within = {}, [], [], [], [], [], {}
    for case in cases:
        ends, p = predictions[case['name']]
        cache = caches[case['name']]
        rows = predictions_to_rows(ends, p, float(cache['fps']), int(cache['count']))
        per_case[case['name']] = clip_metrics(pd.DataFrame(rows), pd.read_csv(case['labels']), case['video'].name)
        y = window_label(ends, case['clips'])
        ok = np.isfinite(y)
        windows_p += list(np.asarray(p)[ok])
        windows_y += list(y[ok])
        own_p, own_y = [], []
        for c in case['clips'].itertuples():
            inside = (ends - .5 >= c.start - 1e-6) & (ends <= c.end + 1e-6)
            if inside.any():
                own_p.append(float(np.mean(np.asarray(p)[inside])))
                own_y.append(1. if c.label == 'locomotion' else 0.)
        clip_p += own_p
        clip_y += own_y
        if len(set(own_y)) == 2:
            within[case['name']] = auc(own_p, own_y)
        known += [r['known_fraction'] for r in per_case[case['name']]['clips']]
    verdict = judge(tally(per_case), CRITERIA)
    verdict.pop('per_video')
    clips = [r for m in per_case.values() for r in m['clips']]
    return dict(verdict=verdict, window_auc=auc(windows_p, windows_y), clip_auc=auc(clip_p, clip_y),
                within_video_clip_auc=within,
                clip_coverage=float(np.mean([r['prediction'] != 'unknown' for r in clips])),
                seconds_coverage=float(np.mean(known)),
                sensitivity_ci=wilson(verdict['locomotion_hits'], verdict['locomotion_clips']),
                specificity_ci=wilson(verdict['resting_correct'], verdict['resting_clips']),
                per_video={n: dict(m['detection'], clips=[dict(clip_id=r['clip_id'], label=r['label'], prediction=r['prediction'],
                                                                 correct=r['detection_correct'], events=len(r['candidate_events']))
                                                            for r in m['clips']]) for n, m in per_case.items()})


def training_set(cases, features, names, kind):
    xs, ys, gs = [], [], []
    for case in cases:
        if case['name'] not in names:
            continue
        data = features[case['name']]
        y = window_label(data['seconds'], case['clips'])
        ok = np.isfinite(y)
        xs.append(feature_set(data, kind)[ok])
        ys.append(y[ok])
        gs += [case['name']] * int(ok.sum())
    return np.concatenate(xs), np.concatenate(ys), np.array(gs)


def youtube_set(labels, feature_dir):
    from .webclips import youtube_windows
    return youtube_windows(labels, feature_dir)


def probe(manifest, cache_dir, feature_dir, out=OUT, youtube_labels=None, youtube_features=None, frozen=()):
    cases = [c for c in load_cases(manifest)]
    caches = {c['name']: load_cache(Path(cache_dir) / f"{c['name']}.npz") for c in cases}
    features = {c['name']: dict(np.load(Path(feature_dir) / f"{c['name']}.npz")) for c in cases}
    youtube = youtube_set(youtube_labels, youtube_features) if youtube_labels and youtube_features else None
    methods = ['lr_full', 'lr_crop', 'lr_full_crop'] + (['lr_full_youtube'] if youtube is not None else [])
    results, chosen = {}, {}
    for grouping in ('video', 'session'):
        group_of = {c['name']: c['name'] if grouping == 'video' else c['session'] for c in cases}
        for method in methods:
            kind = method.split('_', 1)[1].replace('_youtube', '')
            predictions, l2s = {}, {}
            for held in sorted(set(group_of.values())):
                train_names = {n for n, g in group_of.items() if g != held}
                x, y, g = training_set(cases, features, train_names, kind)
                if method == 'lr_full_youtube':
                    yx, yy, _ = youtube
                    l2 = choose_l2(x, y, g)
                    weights = np.r_[balanced(yy), CINNA_WEIGHT * balanced(y) * len(yy) / max(len(y), 1)]
                    model = fit_weighted(np.r_[yx, x], np.r_[yy, y], weights, l2)
                else:
                    l2 = choose_l2(x, y, g)
                    model = fit(x, y, l2)
                l2s[held] = l2
                for case in cases:
                    if group_of[case['name']] == held:
                        data = features[case['name']]
                        predictions[case['name']] = (data['seconds'], predict(model, feature_set(data, kind)))
            results[f'{method}__{grouping}'] = score(cases, predictions, caches)
            chosen[f'{method}__{grouping}'] = l2s
            save_predictions(out, f'{method}__{grouping}', predictions)
            report_line(f'{method}__{grouping}', results[f'{method}__{grouping}'])
    for path in frozen:
        model = json.loads(Path(path).read_text())['model']
        predictions = {c['name']: (features[c['name']]['seconds'], predict(model, features[c['name']]['full'])) for c in cases}
        name = f'youtube_frozen_{Path(path).stem}'
        results[name] = score(cases, predictions, caches)
        save_predictions(out, name, predictions)
        report_line(name, results[name])
    report = dict(results=results, l2_by_fold=chosen, criteria=CRITERIA, band=BAND,
                  protocol=('Línea experimental nueva; no modifica veredictos anteriores. Validación cruzada agrupada '
                            'por vídeo y por sesión (session_of). Estandarización, L2 (validación interna agrupada) y '
                            'pesos se ajustan solo con el entrenamiento de cada pliegue. Ventanas de entrenamiento: '
                            'contenidas entera dentro de clips verificados. Los modelos YouTube congelados son '
                            'referencia, sin reentrenar.'),
                  inputs=dict(manifest_sha256=sha256(manifest), source_sha256=sha256(Path(__file__)),
                              youtube_labels_sha256=sha256(youtube_labels) if youtube_labels else None,
                              python=platform.python_version()))
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / 'probe.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=float))
    return report


def save_predictions(out, name, predictions):
    folder = Path(out) / 'predicciones'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{name}.json').write_text(json.dumps({n: [[round(float(t), 3), round(float(q), 4)] for t, q in zip(*v)]
                                                    for n, v in predictions.items()}))


def load_predictions(out, name):
    raw = json.loads((Path(out) / 'predicciones' / f'{name}.json').read_text())
    return {n: (np.array([r[0] for r in v]), np.array([r[1] for r in v])) for n, v in raw.items()}


def report_line(name, result):
    v = result['verdict']
    print(f"{name:<34} loc {v['locomotion_hits']:>2}/{v['locomotion_clips']:<3} reposo {v['resting_correct']:>2}/"
          f"{v['resting_clips']:<3} falsos {v['resting_false_events']:>2}  AUC clip {result['clip_auc'] or 0:.2f}  "
          f"intra-vídeo {np.mean(list(result['within_video_clip_auc'].values()) or [np.nan]):.2f}  "
          f"cobertura {result['clip_coverage']:.2f}  {'PASA' if v['passed'] else 'no pasa'}", flush=True)


# --- ajuste parcial de X-CLIP ---------------------------------------------------------------------------------

def training_keys(case, cache):
    """Comienzos (cada 0,25 s) de ventanas contenidas en clips verificados, con su etiqueta."""
    items = []
    for c in case['clips'].itertuples():
        for first in range(int(np.ceil(c.start / SPACING - 1e-6)), int(np.floor((c.end - SPAN) / SPACING + 1e-6)) + 1):
            keys = list(range(first, first + FRAMES))
            if all(k in cache['index'] for k in keys):
                items.append((case['name'], keys, 1. if c.label == 'locomotion' else 0.))
    return items


class TwoViewHead:
    def __init__(self, torch, dim, device):
        self.layer = torch.nn.Linear(2 * dim + 1, 1).to(device)

    def __call__(self, torch, full, crop, valid):
        full, crop = torch.nn.functional.normalize(full, dim=-1), torch.nn.functional.normalize(crop, dim=-1)
        return self.layer(torch.cat([full, crop, valid[:, None]], dim=-1)).squeeze(-1)


def finetune(manifest, cache_dir, out=OUT, device='mps', grouping='video', model_id=MODEL_ID, settings=FINETUNE,
             only=None):
    import torch
    cases = load_cases(manifest)
    caches = {c['name']: load_cache(Path(cache_dir) / f"{c['name']}.npz") for c in cases}
    group_of = {c['name']: c['name'] if grouping == 'video' else c['session'] for c in cases}
    name = f'finetune_two_view__{grouping}'
    folder = Path(out) / 'predicciones'
    partial = folder / f'{name}.partial.json'
    predictions = {}
    if partial.is_file():
        raw = json.loads(partial.read_text())
        predictions = {n: (np.array([r[0] for r in v]), np.array([r[1] for r in v])) for n, v in raw.items()}
    for held in sorted(set(group_of.values())):
        test_cases = [c for c in cases if group_of[c['name']] == held]
        if (only and held not in only) or all(c['name'] in predictions for c in test_cases):
            continue
        rng = np.random.default_rng(settings['seed'])
        torch.manual_seed(settings['seed'])
        encoder = Encoder(device, model_id, trainable=True)
        head = TwoViewHead(torch, encoder.model.config.projection_dim, device)
        optimizer = torch.optim.AdamW([dict(params=encoder.parameters(), lr=settings['lr_encoder']),
                                       dict(params=head.layer.parameters(), lr=settings['lr_head'])],
                                      weight_decay=settings['weight_decay'])
        items = [i for c in cases if group_of[c['name']] != held for i in training_keys(c, caches[c['name']])]
        positives = [i for i in items if i[2] == 1.]
        negatives = [i for i in items if i[2] == 0.]
        if not positives or not negatives:
            raise ValueError(f'Pliegue {held}: faltan ventanas de una clase.')
        per_class = max(len(positives), len(negatives))
        for epoch in range(settings['epochs']):
            order = ([positives[i] for i in rng.integers(0, len(positives), per_class)] +
                     [negatives[i] for i in rng.integers(0, len(negatives), per_class)])
            rng.shuffle(order)
            losses = []
            for start in range(0, len(order), settings['batch']):
                chunk = order[start:start + settings['batch']]
                windows = [window_images(caches[n], keys, rng) for n, keys, _ in chunk]
                logits = head(torch, encoder(np.stack([w[0] for w in windows])), encoder(np.stack([w[1] for w in windows])),
                              torch.tensor([w[2] for w in windows], dtype=torch.float32, device=device))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.tensor([y for _, _, y in chunk], dtype=torch.float32, device=device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            print(f'pliegue {held} época {epoch + 1}/{settings["epochs"]} pérdida {np.mean(losses):.3f} '
                  f'({len(positives)} ventanas loc, {len(negatives)} reposo)', flush=True)
        encoder.model.eval()
        for case in test_cases:
            cache = caches[case['name']]
            ends, keys = grid(cache)
            p = []
            with torch.no_grad():
                for start in range(0, len(ends), 16):
                    windows = [window_images(cache, k) for k in keys[start:start + 16]]
                    logits = head(torch, encoder(np.stack([w[0] for w in windows])), encoder(np.stack([w[1] for w in windows])),
                                  torch.tensor([w[2] for w in windows], dtype=torch.float32, device=device))
                    p += torch.sigmoid(logits).float().cpu().numpy().tolist()
            predictions[case['name']] = (ends, np.array(p))
        folder.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps({n: [[round(float(t), 3), round(float(q), 4)] for t, q in zip(*v)]
                                       for n, v in predictions.items()}))
        del encoder, head, optimizer
        if device == 'mps':
            torch.mps.empty_cache()
    if only:
        return None
    save_predictions(out, name, predictions)
    result = score(cases, predictions, caches)
    report_line(name, result)
    path = Path(out) / 'finetune.json'
    report = json.loads(path.read_text()) if path.is_file() else {}
    report[name] = dict(result=result, settings=settings, criteria=CRITERIA, band=BAND,
                        inputs=dict(manifest_sha256=sha256(manifest), source_sha256=sha256(Path(__file__)),
                                    model_id=model_id, torch=torch.__version__, device=device))
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=float))
    return result


# --- diagnóstico ----------------------------------------------------------------------------------------------

def camera_shift(cache, keys):
    """Desplazamiento global mediano entre fotogramas consecutivos (fracción del ancho por 0,25 s)."""
    shifts, previous = [], None
    for k in keys:
        if k not in cache['index']:
            continue
        image = cv2.cvtColor(frame_of(cache, k)[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        if previous is not None and previous.shape == image.shape:
            (dx, dy), _ = cv2.phaseCorrelate(previous, image)
            shifts.append(np.hypot(dx, dy) / image.shape[1])
        previous = image
    return float(np.median(shifts)) if shifts else 0.


def classify_errors(case, cache, metrics):
    rows = []
    clips = case['clips'].sort_values('start').reset_index(drop=True)
    for r in metrics['clips']:
        clip = clips[clips.clip_id == r['clip_id']].iloc[0]
        keys = [k for k in range(int(np.floor(clip.start / SPACING)), int(np.ceil(clip.end / SPACING)) + 1) if k in cache['index']]
        rows_i = [cache['index'][k] for k in keys]
        tracked = np.isfinite(cache['box'][rows_i, 0])
        area = float(np.median(cache['area'][rows_i][tracked])) if tracked.any() else 0.
        centres = cache['centre'][rows_i][tracked]
        span = float(np.linalg.norm(centres.max(axis=0) - centres.min(axis=0)) / max(np.sqrt(np.median(
            cache['box'][rows_i][tracked][:, 2] * cache['box'][rows_i][tracked][:, 3])), 1.)) if tracked.sum() > 1 else 0.
        neighbours = clips[(abs(clips.end - clip.start) < .6) | (abs(clips.start - clip.end) < .6)]
        flags = dict(tracking_loss=float(tracked.mean()) < ERROR_RULES['tracking'], abstention=r['prediction'] == 'unknown',
                     transition=bool((neighbours.label != clip.label).any()),
                     camera_motion=camera_shift(cache, keys) > ERROR_RULES['camera'], small_rabbit=area < ERROR_RULES['small'],
                     short_displacement=clip.label == 'locomotion' and span < ERROR_RULES['short'])
        cause = next((k for k, v in flags.items() if v), 'other') if not r['detection_correct'] else None
        rows.append(dict(case=case['name'], clip_id=r['clip_id'], label=clip.label, start=clip.start, end=clip.end,
                         prediction=r['prediction'], events=len(r['candidate_events']), correct=r['detection_correct'],
                         error=None if r['detection_correct'] else ('missed_locomotion' if clip.label == 'locomotion' else
                                                                    'false_episode' if r['candidate_events'] else 'resting_not_confirmed'),
                         cause=cause, tracked_fraction=float(tracked.mean()) if len(tracked) else 0., area_median=area,
                         centroid_span_body=span, **flags))
    return rows


def render(case, cache, ends, p, out, width=640, fps_out=15.):
    video, (fps, count) = case['video'], metadata(case['video'])
    capture = cv2.VideoCapture(str(video))
    ok, image = capture.read()
    if not ok:
        raise RuntimeError(f'No se puede leer {video}')
    scale = width / image.shape[1]
    size = (width, int(round(image.shape[0] * scale)) + 60)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps_out, size)
    every, duration = max(1, round(fps / fps_out)), count / fps
    colours = dict(locomotion=(60, 180, 255), resting=(200, 160, 60))
    strip = np.full((60, width, 3), 30, np.uint8)
    for c in case['clips'].itertuples():
        cv2.rectangle(strip, (int(c.start / duration * width), 4), (int(c.end / duration * width), 24), colours[c.label], -1)
    for t, q in zip(ends, p):
        colour = (60, 180, 255) if q >= BAND[1] else (200, 160, 60) if q <= BAND[0] else (120, 120, 120)
        cv2.rectangle(strip, (int((t - .5) / duration * width), 32), (int(t / duration * width), 56), colour, -1)
    try:
        frame = 0
        while ok:
            if frame % every == 0:
                t = frame / fps
                small = cv2.resize(image, (width, size[1] - 60), interpolation=cv2.INTER_AREA)
                k = int(round(t / SPACING))
                if k in cache['index'] and np.isfinite(cache['box'][cache['index'][k], 0]):
                    x, y, w, h = cache['box'][cache['index'][k]] * width / (CACHE_SIDE if image.shape[1] >= image.shape[0]
                                                                         else CACHE_SIDE * image.shape[1] / image.shape[0])
                    cv2.rectangle(small, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                current = [q for e, q in zip(ends, p) if e - .5 < t <= e]
                label = next((c.label for c in case['clips'].itertuples() if c.start <= t < c.end), '-')
                text = f't={t:5.1f}s  verdad={label}  p_loc={current[0]:.2f}' if current else f't={t:5.1f}s  verdad={label}  p_loc=n/a'
                cv2.putText(small, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
                bottom = strip.copy()
                cv2.line(bottom, (int(t / duration * width), 0), (int(t / duration * width), 59), (255, 255, 255), 1)
                writer.write(np.vstack([small, bottom]))
            ok, image = capture.read()
            frame += 1
    finally:
        capture.release()
        writer.release()


def diagnose(manifest, cache_dir, out=OUT, method=None, videos=None):
    out = Path(out)
    if method is None:
        report = json.loads((out / 'probe.json').read_text())['results']
        if (out / 'finetune.json').is_file():
            report.update({k: v['result'] for k, v in json.loads((out / 'finetune.json').read_text()).items()})
        method = max((k for k in report if k.endswith('__video')),
                     key=lambda k: (min(report[k]['verdict']['sensitivity'], report[k]['verdict']['specificity']),
                                    -report[k]['verdict']['resting_false_events']))
    predictions = load_predictions(out, method)
    cases = load_cases(manifest)
    rows = []
    folder = Path(videos) if videos else REPO / 'runs' / 'adapt' / 'diagnostico' / method
    folder.mkdir(parents=True, exist_ok=True)
    for case in cases:
        cache = load_cache(Path(cache_dir) / f"{case['name']}.npz")
        ends, p = predictions[case['name']]
        metrics = clip_metrics(pd.DataFrame(predictions_to_rows(ends, p, float(cache['fps']), int(cache['count']))),
                               pd.read_csv(case['labels']), case['video'].name)
        rows += classify_errors(case, cache, metrics)
        if case['video'].is_file():
            render(case, cache, ends, p, folder / f"{case['name']}.mp4")
    table = pd.DataFrame(rows)
    errors = table[~table.correct]
    summary = dict(method=method, clips=len(table), errors=len(errors),
                   by_error=errors.error.value_counts().to_dict(), by_cause=errors.cause.value_counts().to_dict(),
                   by_error_and_cause=errors.groupby(['error', 'cause']).size().rename('n').reset_index().to_dict('records'),
                   rules=ERROR_RULES, order=['tracking_loss', 'abstention', 'transition', 'camera_motion', 'small_rabbit',
                                              'short_displacement', 'other'],
                   videos=str(folder))
    table.to_csv(out / f'errores__{method}.csv', index=False)
    (out / f'errores__{method}.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    return summary


def main():
    parser = argparse.ArgumentParser(description='Adaptación experimental del modelo temporal a Cinna')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('audit', 'cache', 'embed', 'probe', 'finetune', 'diagnose'):
        command = sub.add_parser(name)
        command.add_argument('--manifest', required=True)
        command.add_argument('--out', default=str(OUT))
        if name != 'audit':
            command.add_argument('--cache', default=str(REPO / 'runs' / 'adapt' / 'cache'))
    sub.choices['audit'].add_argument('--features')
    sub.choices['audit'].add_argument('--cache')
    for name in ('embed', 'finetune'):
        sub.choices[name].add_argument('--device', choices=['cpu', 'mps'], default='mps')
        sub.choices[name].add_argument('--model-id', default=MODEL_ID)
    sub.choices['embed'].add_argument('--features', default=str(REPO / 'runs' / 'adapt' / 'features'))
    sub.choices['embed'].add_argument('--reference', help='Características de webclips para comprobar la vista completa')
    sub.choices['probe'].add_argument('--features', default=str(REPO / 'runs' / 'adapt' / 'features'))
    sub.choices['probe'].add_argument('--youtube-labels')
    sub.choices['probe'].add_argument('--youtube-features')
    sub.choices['probe'].add_argument('--frozen', action='append', default=[])
    sub.choices['finetune'].add_argument('--grouping', choices=['video', 'session'], default='video')
    sub.choices['finetune'].add_argument('--only', action='append')
    sub.choices['diagnose'].add_argument('--method')
    args = parser.parse_args()
    if args.command == 'audit':
        audit(args.manifest, args.features, args.cache, args.out)
    elif args.command == 'cache':
        for case in load_cases(args.manifest):
            print('CACHÉ', case['name'], cache_case(case, args.cache), flush=True)
    elif args.command == 'embed':
        embed(args.manifest, args.cache, args.features, Encoder(args.device, args.model_id), reference=args.reference)
    elif args.command == 'probe':
        probe(args.manifest, args.cache, args.features, args.out, args.youtube_labels, args.youtube_features, args.frozen)
    elif args.command == 'finetune':
        finetune(args.manifest, args.cache, args.out, args.device, args.grouping, args.model_id, only=args.only)
    else:
        diagnose(args.manifest, args.cache, args.out, args.method)


if __name__ == '__main__':
    main()
