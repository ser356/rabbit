import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import adapt, webclips
from .motion import sha256


CONFIGS = tuple(itertools.product(('appearance', 'motion'), ('youtube', 'combined'), (.01, .1, 1.), ('window', 'mil')))
SMOOTHING = (1, 3)


def encode_contrast(encoder, windows):
    with encoder.torch.no_grad():
        real = encoder(windows).float().cpu().numpy()
        still = encoder(np.repeat(windows[:, 4:5], webclips.FRAMES, axis=1)).float().cpu().numpy()
    real /= np.linalg.norm(real, axis=1, keepdims=True)
    still /= np.linalg.norm(still, axis=1, keepdims=True)
    return real, still


def extract(video, seconds, encoder, output):
    fps, count = webclips.metadata(video)
    keys = {float(t): webclips.window_keys(t) for t in seconds}
    frames = webclips.sample_frames(video, {k for ks in keys.values() if ks for k in ks})
    ends = np.array([t for t, ks in keys.items() if ks and all(k in frames for k in ks)])
    real, still = [], []
    for start in range(0, len(ends), 16):
        windows = np.stack([np.stack([frames[k] for k in keys[t]]) for t in ends[start:start + 16]])
        a, b = encode_contrast(encoder, windows)
        real.append(a)
        still.append(b)
    first = np.array([round(keys[t][0] * webclips.SPACING * fps) / fps for t in ends])
    last = np.array([round(keys[t][-1] * webclips.SPACING * fps) / fps for t in ends])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, seconds=ends, appearance=np.concatenate(real), still=np.concatenate(still),
             start=first, stop=last, fps=fps, count=count, video_sha256=sha256(video))
    print('CONTRASTE', video.stem, len(ends), flush=True)


def embed(manifest, labels, youtube_features, folders, out, device):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'encoder.json').unlink(missing_ok=True)
    encoder = adapt.Encoder(device)
    for case in adapt.load_cases(manifest):
        fps, count = webclips.metadata(case['video'])
        extract(case['video'], webclips.window_ends(count, fps), encoder, out / 'cinna' / f"{case['name']}.npz")
    videos = {}
    for folder in folders:
        for path in webclips.videos_in(folder):
            if path.stem in videos:
                raise ValueError(f'Vídeo duplicado: {path.stem}')
            videos[path.stem] = path
    table = pd.read_csv(labels)
    for name in sorted(table.loc[table.label.isin(adapt.CLASSES), 'video'].unique()):
        reference = np.load(Path(youtube_features) / f'{name}.npz')
        extract(videos[name], reference['seconds'], encoder, out / 'youtube' / f'{name}.npz')
    (out / 'encoder.json').write_text(json.dumps(dict(**encoder.info, source_sha256=sha256(Path(__file__))), indent=2))


def labels_inside(data, clips):
    y = np.full(len(data['seconds']), np.nan)
    data['bag'] = np.full(len(y), -1, dtype=int)
    for i, c in enumerate(clips.itertuples()):
        inside = (data['start'] >= c.start - 1e-6) & (data['stop'] <= c.end + 1e-6)
        if np.isfinite(y[inside]).any():
            raise ValueError('Clips de entrenamiento solapados')
        y[inside] = float(c.label == 'locomotion')
        data['bag'][inside] = i
    return y


def representation(data, kind):
    if kind == 'motion':
        return np.abs(data['appearance'] - data['still'])
    return data[kind]


def group_weights(y, groups):
    weights = np.zeros(len(y))
    for label in (0., 1.):
        names = np.unique(groups[y == label])
        if not len(names):
            raise ValueError('Falta una clase en entrenamiento')
        for name in names:
            mask = (groups == name) & (y == label)
            weights[mask] = 1. / (2 * len(names) * mask.sum())
    return weights


def fit_model(training, external, config):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    kind, source, strength, loss = config
    datasets = [external] if source == 'youtube' else [training] if source == 'cinna' else [external, training]
    xs, ys, weights, bags, bag_y, bag_weights = [], [], [], [], [], []
    for entries in datasets:
        x, y, groups, local_bag_y, local_bag_groups = [], [], [], [], []
        for name, data, labels in entries:
            valid = np.isfinite(labels)
            x.append(representation(data, kind)[valid])
            y.extend(labels[valid])
            groups.extend([name] * int(valid.sum()))
            if loss.endswith('mil'):
                ids = data['bag'][valid]
                for bag in np.unique(ids):
                    local_bag_y.append(float(labels[valid][ids == bag][0]))
                    local_bag_groups.append(name)
                offset = len(bag_y) + len(local_bag_y) - len(np.unique(ids))
                bags.extend(offset + np.searchsorted(np.unique(ids), ids))
        xs.append(np.concatenate(x))
        ys.append(np.array(y))
        if loss.endswith('mil'):
            bw = group_weights(np.array(local_bag_y), np.array(local_bag_groups)) / len(datasets)
            ids = np.array(bags[-len(y):]) - len(bag_y)
            weights.append(bw[ids] / np.bincount(ids)[ids])
            bag_y.extend(local_bag_y)
            bag_weights.extend(bw)
        else:
            weights.append(group_weights(np.array(y), np.array(groups)) / len(datasets))
    x, y = np.concatenate(xs), np.concatenate(ys)
    weight = np.concatenate(weights) * len(y)
    varying = np.ptp(x, axis=0) > 1e-12
    if not varying.any():
        raise ValueError('Ninguna característica varía en entrenamiento')
    scaler = StandardScaler().fit(x[:, varying], sample_weight=weight)
    z = scaler.transform(x[:, varying])
    basis = None
    if loss.startswith('rbf_'):
        from sklearn.kernel_approximation import RBFSampler

        basis = RBFSampler(gamma=1 / z.shape[1], n_components=128, random_state=0)
        z = basis.fit_transform(z)
    if loss.endswith('mil'):
        import torch

        torch.set_num_threads(1)
        z = torch.tensor(z, dtype=torch.float32)
        ids = torch.tensor(bags, dtype=torch.long)
        target = torch.tensor(bag_y, dtype=torch.float32)
        bw = torch.tensor(bag_weights, dtype=torch.float32)
        w = torch.zeros(z.shape[1], requires_grad=True)
        intercept = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.Adam([w, intercept], lr=.03)
        for _ in range(300):
            logits = z @ w + intercept
            pooled = torch.full((len(bag_y),), -torch.inf).scatter_reduce(0, ids, logits, reduce='amax')
            objective = (torch.nn.functional.binary_cross_entropy_with_logits(pooled, target, reduction='none') * bw).sum()
            objective = objective + w.square().sum() / (2 * strength * len(bag_y))
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
        coefficients = np.r_[intercept.detach().numpy(), w.detach().numpy()]
    else:
        model = LogisticRegression(C=strength, max_iter=2000, solver='lbfgs', random_state=0)
        model.fit(z, y, sample_weight=weight)
        coefficients = np.r_[model.intercept_, model.coef_[0]]
    mean, scale, expanded = x[0].copy(), np.ones(x.shape[1]), np.zeros(x.shape[1] + 1)
    mean[varying], scale[varying], expanded[0] = scaler.mean_, scaler.scale_, coefficients[0]
    if basis is not None:
        projection = np.zeros((x.shape[1], 128))
        projection[varying] = basis.random_weights_
        return dict(mean=mean.tolist(), scale=scale.tolist(), weights=coefficients.tolist(),
                    projection=projection.tolist(), offset=basis.random_offset_.tolist())
    expanded[1:][varying] = coefficients[1:]
    return dict(mean=mean.tolist(), scale=scale.tolist(), weights=expanded.tolist())


def predict_model(model, x):
    if 'projection' not in model:
        return adapt.predict(model, x)
    z = (np.asarray(x) - model['mean']) / model['scale']
    z = np.sqrt(2 / len(model['offset'])) * np.cos(z @ np.array(model['projection']) + model['offset'])
    weights = np.array(model['weights'])
    return 1 / (1 + np.exp(-np.clip(weights[0] + z @ weights[1:], -30, 30)))


def smooth_predictions(predictions, width):
    result = {}
    for name, (seconds, p) in predictions.items():
        q = np.array(p, copy=True)
        for left, right in zip(np.r_[0, np.flatnonzero(np.diff(seconds) > .75) + 1],
                               np.r_[np.flatnonzero(np.diff(seconds) > .75) + 1, len(seconds)]):
            for i in range(left, right):
                q[i] = np.mean(p[max(left, i - width // 2):min(right, i + width // 2 + 1)])
        result[name] = (seconds, q)
    return result


def selection_key(result):
    v = result['verdict']
    return v['passed'], min(v['sensitivity'], v['specificity']), -v['resting_false_events'], v['sensitivity'] + v['specificity']


def nested(cases, data, external, out, configs=CONFIGS, smoothing=SMOOTHING):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    labels = {c['name']: labels_inside(data[c['name']], c['clips']) for c in cases}
    sessions = sorted({c['session'] for c in cases})
    fitted = {}

    def fit(entries, config):
        key = (config, tuple(name for name, _, _ in entries) if config[1] != 'youtube' else ())
        if key not in fitted:
            fitted[key] = fit_model(entries, external, config)
        return fitted[key]

    selected, choices, ablations = {}, {}, {(cfg, width): {} for cfg in configs for width in smoothing}
    for held in sessions:
        training_cases = [c for c in cases if c['session'] != held]
        test_cases = [c for c in cases if c['session'] == held]
        training = [(c['name'], data[c['name']], labels[c['name']]) for c in training_cases]
        scores, models = {}, {}
        for cfg in configs:
            inner = {}
            for validation in sorted({c['session'] for c in training_cases}):
                entries = [(c['name'], data[c['name']], labels[c['name']]) for c in training_cases
                           if c['session'] != validation]
                model = fit(entries, cfg)
                for case in training_cases:
                    if case['session'] == validation:
                        d = data[case['name']]
                        inner[case['name']] = (d['seconds'], predict_model(model, representation(d, cfg[0])))
            models[cfg] = fit(training, cfg)
            predictions = {c['name']: (data[c['name']]['seconds'], predict_model(models[cfg], representation(data[c['name']], cfg[0])))
                           for c in test_cases}
            for width in smoothing:
                key = (cfg, width)
                scores[key] = adapt.score(training_cases, smooth_predictions(inner, width), data)
                ablations[key].update(smooth_predictions(predictions, width))
        best = max(scores, key=lambda key: selection_key(scores[key]))
        selected.update({c['name']: ablations[best][c['name']] for c in test_cases})
        choices[held] = dict(config=list(best[0]), smoothing=best[1], training_sessions=[s for s in sessions if s != held],
                             inner_scores={str(k): {n: v for n, v in r.items() if n != 'per_video'} for k, r in scores.items()},
                             model=models[best[0]])
        print('SESIÓN', held, best, flush=True)
    result = adapt.score(cases, selected, data)
    adapt.save_predictions(out, 'selected_nested__session', selected)
    adapt.report_line('selected_nested__session', result)
    results = {}
    for (config, width), predictions in ablations.items():
        name = '_'.join(map(str, (*config, width)))
        results[name] = adapt.score(cases, predictions, data)
        adapt.save_predictions(out, name, predictions)
        adapt.report_line(name, results[name])
    report = dict(result=result, choices=choices, ablations=results, criteria=adapt.CRITERIA, band=webclips.BAND,
                  status='development_criteria_met' if result['verdict']['passed'] else 'not_resolved',
                  protocol='Validación anidada por sesión. Datos reutilizados de desarrollo; no test independiente.')
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return report


def probe(manifest, labels, features, out, fusion=False, nonlinear=False, extended=False):
    features, out = Path(features), Path(out)
    cases = adapt.load_cases(manifest)
    data = {c['name']: dict(np.load(features / 'cinna' / f"{c['name']}.npz")) for c in cases}
    external = []
    table = pd.read_csv(labels)
    for name, clips in table[table.label.isin(adapt.CLASSES)].groupby('video'):
        d = dict(np.load(features / 'youtube' / f'{name}.npz'))
        clips = clips.rename(columns={'start_s': 'start', 'end_s': 'end'})
        external.append((name, d, labels_inside(d, clips)))
    physical_paths = []
    configs = CONFIGS
    if fusion:
        models = {(kind, loss): fit_model([], external, (kind, 'youtube', .1, loss))
                  for kind in ('appearance', 'motion') for loss in ('window', 'mil')}
        fields = ['speed_bodywidths_s', 'speed_lower', 'speed_upper', 'valid_fraction', 'evidence_quality',
                  'camera_error_px', 'mask_fraction', 'elongation_range', 'log_area_range']
        entries = json.loads(Path(manifest).read_text())['cases']
        for case in entries:
            path = (Path(manifest).parent / Path(case['floor']).expanduser()).resolve()
            physical_paths.append(path)
            frame = pd.read_csv(path)
            frame.index = frame.seconds.round(3)
            d = data[case['name']]
            x = frame.loc[np.round(d['seconds'], 3), fields].to_numpy(float)
            if np.any(x[np.isfinite(x)] < 0):
                raise ValueError('Medida física negativa inesperada')
            d['physical'] = np.c_[np.log1p(np.where(np.isfinite(x), x, 0.)), ~np.isfinite(x)]
            p = np.column_stack([predict_model(m, representation(d, k)) for (k, _), m in models.items()])
            d['fusion'] = np.c_[d['physical'], p]
        losses = ('window', 'mil', 'rbf_window', 'rbf_mil') if nonlinear else ('window', 'mil')
        configs = tuple(itertools.product(('physical', 'fusion'), ('cinna',), (.01, .1, 1.), losses))
        if extended:
            configs += tuple(itertools.product(('physical', 'fusion'), ('cinna',), (10., 100.), ('rbf_window', 'rbf_mil')))
    report = nested(cases, data, external, out, configs=configs)
    files = [Path(manifest), Path(labels), Path(__file__), Path(adapt.__file__), Path(webclips.__file__),
             features / 'encoder.json'] + [c['labels'] for c in cases] + sorted(features.glob('*/*.npz')) + physical_paths
    report['inputs'] = {str(p): sha256(p) for p in files}
    report['source'] = Path(__file__).read_text()
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('embed', 'probe', 'fuse', 'nonlinear', 'extended'))
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--out', default=str(adapt.REPO / 'cierre/adapt_motion'))
    parser.add_argument('--youtube-features')
    parser.add_argument('--youtube-dir', action='append', default=[])
    parser.add_argument('--device', default='mps', choices=('cpu', 'mps'))
    args = parser.parse_args()
    if args.command == 'embed':
        embed(args.manifest, args.labels, args.youtube_features, args.youtube_dir, args.features, args.device)
    else:
          probe(args.manifest, args.labels, args.features, args.out, fusion=args.command in ('fuse', 'nonlinear', 'extended'),
              nonlinear=args.command in ('nonlinear', 'extended'), extended=args.command == 'extended')


if __name__ == '__main__':
    main()