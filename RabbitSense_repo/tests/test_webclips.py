import json

import cv2
import numpy as np
import pandas as pd
import pytest

from rabbitsense import webclips


def make_video(path, frames=150, fps=10., size=(96, 64), moving=True):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for f in range(frames):
        image = np.full((size[1], size[0], 3), 60, np.uint8)
        x = (10 + 2 * f) % (size[0] - 20) if moving else 30
        image[20:40, x:x + 20] = 220
        writer.write(image)
    writer.release()
    return path


def test_sheets_pick_spread_clips_and_write_rows(tmp_path):
    folder = tmp_path / 'yt'
    folder.mkdir()
    make_video(folder / 'abc.mp4', frames=500)
    make_video(folder / 'short.mp4', frames=30)
    (folder / 'notes.txt').write_text('x')
    chosen = webclips.chosen_clips(50.)
    assert len(chosen) == 8 and chosen[0] == 0. and chosen[-1] == 44. and chosen == sorted(set(chosen))
    assert webclips.sheets(folder, tmp_path / 'sheets') == 8
    clips = pd.read_csv(tmp_path / 'sheets' / 'clips.csv')
    assert set(clips.video) == {'abc'} and clips.start_s.iloc[0] == 0. and (clips.end_s - clips.start_s == 4.).all()
    sheet = cv2.imread(str(tmp_path / 'sheets' / 'abc.jpg'))
    assert sheet.shape[0] == 8 * webclips.THUMB and sheet.shape[1] == 70 + 8 * webclips.THUMB


def test_letterbox_keeps_whole_frame():
    image = np.zeros((300, 100, 3), np.uint8)
    image[:, :, 2] = 255
    boxed = webclips.letterbox(image, 60)
    assert boxed.shape == (60, 60, 3)
    assert boxed[:, 20:40, 0].min() > 200 and boxed[:, :5].max() == 0


def test_window_keys_and_grid():
    assert webclips.window_keys(1.) is None
    assert webclips.window_keys(1.25) == list(range(0, 8))
    assert webclips.window_ends(21, 10.) == [0., .5, 1., 1.5, 2.]


def test_logistic_regression_and_grouped_training(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / 'f').mkdir()
    rows = []
    for v in range(10):
        moving = v % 2 == 0
        seconds = np.arange(1.5, 4.0, .5)
        features = rng.normal(0, 1, (len(seconds), 6)) + (1.5 if moving else -1.5) * np.r_[1, 0, 0, 0, 0, 0]
        np.savez(tmp_path / 'f' / f'v{v}.npz', seconds=seconds, features=features)
        rows.append(dict(video=f'v{v}', start_s=0., end_s=4., label='locomotion' if moving else 'resting'))
    rows.append(dict(video='v1', start_s=8., end_s=12., label='unknown'))
    pd.DataFrame(rows).to_csv(tmp_path / 'labels.csv', index=False)
    x, y, groups = webclips.youtube_windows(tmp_path / 'labels.csv', tmp_path / 'f')
    assert len(y) == 10 * 5 and set(groups) == {f'v{v}' for v in range(10)}
    report = webclips.train(tmp_path / 'labels.csv', tmp_path / 'f', tmp_path / 'model.json')
    assert report['grouped_cross_validation']['auc'] > .9 and report['videos'] == 10
    model = json.loads((tmp_path / 'model.json').read_text())['model']
    assert webclips.predict(model, np.r_[3., 0, 0, 0, 0, 0][None])[0] > .9


def test_classify_band_and_missing_windows():
    model = dict(mean=[0.], scale=[1.], weights=[0., 1.])
    rows = webclips.classify([0., .5, 1., 1.5, 2.], np.array([1.5, 2.]), np.array([[3.], [-3.]]), model)
    assert [r['movement'] for r in rows] == ['unknown', 'unknown', 'unknown', 'locomotion_candidate',
                                             'low_motion_candidate']


def test_test_run_on_cinna_uses_frozen_model(tmp_path):
    make_video(tmp_path / 'cinna.mp4', frames=81)
    seconds = np.arange(1.5, 8.01, .5)
    features = np.where(seconds <= 4., 3., -3.)[:, None]
    (tmp_path / 'f').mkdir()
    np.savez(tmp_path / 'f' / 'cinna.npz', seconds=seconds, features=features)
    (tmp_path / 'labels.csv').write_text('clip_id,video_id,reviewed,label,start_seconds,duration_seconds\n'
                                         'a,cinna.mp4,si,locomotion,0,4\nb,cinna.mp4,si,resting,4,4\n')
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=[dict(name='cinna', video='cinna.mp4', labels='labels.csv')])))
    with pytest.raises(SystemExit, match='congelado'):
        webclips.test(tmp_path / 'm.json', tmp_path / 'f', tmp_path / 'out', tmp_path / 'missing.json')
    (tmp_path / 'model.json').write_text(json.dumps(dict(model=dict(mean=[0.], scale=[1.], weights=[0., 1.]))))
    result = webclips.test(tmp_path / 'm.json', tmp_path / 'f', tmp_path / 'out', tmp_path / 'model.json')
    counts = result['methods']['webclips']
    assert (counts['locomotion_hits'], counts['resting_correct'], counts['resting_false_events']) == (1, 1, 0)
    assert result['status'] == 'not_resolved'


def test_embed_with_tiny_random_xclip(tmp_path):
    for module in ('torch', 'torchvision', 'PIL'):
        pytest.importorskip(module)
    transformers = pytest.importorskip('transformers')
    config = transformers.XCLIPConfig(
        text_config=dict(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, max_position_embeddings=77),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                           image_size=224, patch_size=32, num_frames=8, mit_hidden_size=32, mit_intermediate_size=64,
                           mit_num_hidden_layers=1, mit_num_attention_heads=2),
        projection_dim=32, prompt_layers=1, prompt_attention_heads=2)
    model_dir = tmp_path / 'tiny'
    transformers.XCLIPModel(config).save_pretrained(model_dir)
    vocab = {chr(i): i - 33 for i in range(33, 127)}
    vocab.update({'<|startoftext|>': 94, '<|endoftext|>': 95})
    transformers.XCLIPProcessor(image_processor=transformers.VideoMAEImageProcessor(),
                                tokenizer=transformers.CLIPTokenizer(vocab=vocab, merges=[])).save_pretrained(model_dir)
    embedder = webclips.XClipEmbedder('cpu', str(model_dir))
    folder = tmp_path / 'yt'
    folder.mkdir()
    make_video(folder / 'abc.mp4', frames=60)
    pd.DataFrame([dict(video='abc', start_s=0., end_s=4., label='locomotion')]).to_csv(tmp_path / 'l.csv', index=False)
    webclips.embed(folder, tmp_path / 'l.csv', tmp_path / 'out', embedder)
    data = np.load(tmp_path / 'out' / 'abc.npz')
    assert data['features'].shape == (len(data['seconds']), 32)
    assert list(data['seconds']) == [1.5, 2., 2.5, 3., 3.5]
