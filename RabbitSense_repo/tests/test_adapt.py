import json

import cv2
import numpy as np
import pandas as pd
import pytest

from rabbitsense import adapt
from rabbitsense.motion import sha256


def make_case(folder, name, frames=100, fps=10., moving_until=40, size=(160, 96)):
    """Vídeo con un 'conejo' que se desplaza hasta `moving_until` y después queda quieto, más sus máscaras."""
    video = folder / f'{name}.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    masks = folder / f'masks_{name}' / 'masks'
    masks.mkdir(parents=True)
    for f in range(frames):
        image = np.full((size[1], size[0], 3), 70, np.uint8)
        image[:, ::16] = 120
        x = 10 + 2 * min(f, moving_until)
        image[40:60, x:x + 24] = 230
        mask = np.zeros((size[1], size[0]), np.uint8)
        mask[40:60, x:x + 24] = 255
        writer.write(image)
        cv2.imwrite(str(masks / f'{f:06d}.png'), mask)
    writer.release()
    (masks.parent / 'summary.json').write_text(json.dumps(dict(frames=frames, inputs=dict(video=dict(sha256=sha256(video))))))
    return video, masks.parent


def make_manifest(tmp_path, names=('a', 'b', 'c', 'IMG_5190', 'IMG_5191')):
    rows, cases = [], []
    for name in names:
        video, masks = make_case(tmp_path, name)
        rows += [dict(clip_id=f'{name}_0', video_id=video.name, start_seconds=0., duration_seconds=4., label='locomotion',
                      reviewed='si'),
                 dict(clip_id=f'{name}_6', video_id=video.name, start_seconds=6., duration_seconds=4., label='resting',
                      reviewed='si'),
                 dict(clip_id=f'{name}_x', video_id=video.name, start_seconds=4., duration_seconds=2., label='unknown',
                      reviewed='si')]
        cases.append(dict(name=name, video=video.name, labels='labels.csv', masks=masks.name))
    pd.DataFrame(rows).to_csv(tmp_path / 'labels.csv', index=False)
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=cases)))
    return tmp_path / 'm.json'


def tiny_model(tmp_path):
    transformers = pytest.importorskip('transformers')
    config = transformers.XCLIPConfig(
        text_config=dict(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, max_position_embeddings=77),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                           image_size=224, patch_size=32, num_frames=8, mit_hidden_size=32, mit_intermediate_size=64,
                           mit_num_hidden_layers=1, mit_num_attention_heads=2),
        projection_dim=32, prompt_layers=1, prompt_attention_heads=2)
    transformers.XCLIPModel(config).save_pretrained(tmp_path / 'tiny')
    return str(tmp_path / 'tiny')


def test_sessions_and_independence_counts(tmp_path):
    assert adapt.session_of('WhatsApp Video 2026-10-26 at 18.44.35') == adapt.session_of('WhatsApp Video x')
    assert adapt.session_of('IMG_5190') == adapt.session_of('IMG_5193') != adapt.session_of('5177')
    assert adapt.windows_inside(0., 4., .5) == 5 and adapt.windows_inside(0., 4., adapt.SPAN + adapt.SPACING) == 2
    assert adapt.windows_inside(0., 1., .5) == 0
    low, high = adapt.wilson(20, 25)
    assert low < .8 < high and adapt.clips_needed(.9) > 25
    report = adapt.audit(make_manifest(tmp_path), out=tmp_path / 'out')
    totals = report['totals']
    assert (totals['locomotion_clips'], totals['resting_clips'], totals['videos_with_both']) == (5, 5, 5)
    assert totals['sessions'] == 4 and totals['overlapping_windows'] == 50 and totals['disjoint_windows'] == 20
    assert (tmp_path / 'out' / 'auditoria.json').is_file()


def test_window_labels_need_whole_window_inside_a_clip():
    clips = pd.DataFrame(dict(clip_id=['a', 'b'], start=[0., 6.], end=[4., 10.], label=['locomotion', 'resting']))
    labels = adapt.window_label(np.array([1., 1.25, 3., 3.5, 4., 7.5, 10.]), clips)
    assert np.isnan(labels[0]) and labels[1] == 1. and labels[2] == 1. and labels[3] == 1.
    assert np.isnan(labels[4]) and labels[5] == 0. and np.isnan(labels[6])


def test_cache_crops_follow_the_mask(tmp_path):
    manifest = make_manifest(tmp_path, names=('a',))
    case = adapt.load_cases(manifest)[0]
    assert len(case['clips']) == 2
    assert adapt.cache_case(case, tmp_path / 'cache') == 40
    cache = adapt.load_cache(tmp_path / 'cache' / 'a.npz')
    assert np.isfinite(cache['box']).all() and cache['area'].min() > 0
    ends, keys = adapt.grid(cache)
    assert ends[0] == 1.5 and keys[0] == list(range(1, 9))
    full, crop, valid = adapt.window_images(cache, keys[0])
    assert full.shape == crop.shape == (8, 224, 224, 3) and valid == 1.
    assert crop.mean() > full.mean()  # el recorte se centra en el conejo claro
    image = cv2.cvtColor(adapt.frame_of(cache, 0)[0], cv2.COLOR_BGR2GRAY)
    with pytest.raises(ValueError, match='no corresponden'):
        broken = dict(case, masks=tmp_path / 'masks_a')
        (tmp_path / 'masks_a' / 'summary.json').write_text(json.dumps(dict(inputs=dict(video=dict(sha256='x')))))
        adapt.cache_case(broken, tmp_path / 'cache2')
    assert image.shape[1] == adapt.CACHE_SIDE


def test_grouped_probe_has_no_leakage_and_scores(tmp_path):
    manifest = make_manifest(tmp_path)
    cases = adapt.load_cases(manifest)
    (tmp_path / 'f').mkdir()
    rng = np.random.default_rng(0)
    for case in cases:
        adapt.cache_case(case, tmp_path / 'cache')
        cache = adapt.load_cache(tmp_path / 'cache' / f"{case['name']}.npz")
        ends, _ = adapt.grid(cache)
        signal = np.where(ends <= 4.5, 5., -5.)[:, None]
        np.savez(tmp_path / 'f' / f"{case['name']}.npz", seconds=ends, crop_valid=np.ones(len(ends)),
                 full=rng.normal(0, 1, (len(ends), 4)) + signal * np.r_[1, 0, 0, 0],
                 crop=rng.normal(0, 1, (len(ends), 4)))
    seen = []
    original = adapt.training_set

    def spy(cases_, features, names, kind):
        seen.append(set(names))
        return original(cases_, features, names, kind)
    adapt.training_set = spy
    try:
        report = adapt.probe(manifest, tmp_path / 'cache', tmp_path / 'f', tmp_path / 'out')
    finally:
        adapt.training_set = original
    assert all(len(s) in (3, 4) for s in seen)  # nunca el vídeo o la sesión evaluada
    assert not any({'IMG_5190', 'IMG_5191'} & s and not {'IMG_5190', 'IMG_5191'} <= s for s in seen[15:])
    full = report['results']['lr_full__video']
    assert full['verdict']['locomotion_hits'] == 5 and full['verdict']['resting_correct'] == 5 and full['clip_auc'] == 1.
    assert report['results']['lr_crop__video']['clip_auc'] < 1.
    assert set(report['results']) == {f'{m}__{g}' for m in ('lr_full', 'lr_crop', 'lr_full_crop') for g in ('video', 'session')}
    predictions = adapt.load_predictions(tmp_path / 'out', 'lr_full__video')
    assert set(predictions) == {c['name'] for c in cases}


def test_finetune_and_diagnose_with_tiny_random_xclip(tmp_path):
    for module in ('torch', 'transformers'):
        pytest.importorskip(module)
    manifest = make_manifest(tmp_path, names=('a', 'b', 'c'))
    for case in adapt.load_cases(manifest):
        adapt.cache_case(case, tmp_path / 'cache')
    model = tiny_model(tmp_path)
    encoder = adapt.Encoder('cpu', model)
    adapt.embed(manifest, tmp_path / 'cache', tmp_path / 'f', encoder)
    data = np.load(tmp_path / 'f' / 'a.npz')
    assert data['full'].shape == data['crop'].shape == (len(data['seconds']), 32) and data['crop_valid'].min() == 1.
    settings = dict(adapt.FINETUNE, epochs=1, batch=4)
    trainable = adapt.Encoder('cpu', model, trainable=True)
    names = {n for n, p in trainable.model.named_parameters() if p.requires_grad}
    assert names and all(n.startswith(('vision_model.encoder.layers.1.', 'vision_model.post_layernorm', 'visual_projection',
                                       'mit.')) for n in names)
    adapt.finetune(manifest, tmp_path / 'cache', tmp_path / 'out', 'cpu', 'video', model, settings, only=['a'])
    partial = json.loads((tmp_path / 'out' / 'predicciones' / 'finetune_two_view__video.partial.json').read_text())
    assert set(partial) == {'a'}
    result = adapt.finetune(manifest, tmp_path / 'cache', tmp_path / 'out', 'cpu', 'video', model, settings)
    assert result['verdict']['locomotion_clips'] == 3 and 0. <= result['clip_coverage'] <= 1.
    report = json.loads((tmp_path / 'out' / 'finetune.json').read_text())
    assert report['finetune_two_view__video']['settings']['epochs'] == 1
    summary = adapt.diagnose(manifest, tmp_path / 'cache', tmp_path / 'out', 'finetune_two_view__video', tmp_path / 'videos')
    assert summary['clips'] == 6 and (tmp_path / 'videos' / 'a.mp4').is_file()
    errors = pd.read_csv(tmp_path / 'out' / 'errores__finetune_two_view__video.csv')
    assert set(errors.columns) >= {'error', 'cause', 'tracked_fraction', 'camera_motion', 'short_displacement'}
    assert errors[errors.label == 'locomotion'].centroid_span_body.min() > 1.  # el conejo sintético se desplaza
    capture = cv2.VideoCapture(str(tmp_path / 'videos' / 'a.mp4'))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) > 10
    capture.release()
