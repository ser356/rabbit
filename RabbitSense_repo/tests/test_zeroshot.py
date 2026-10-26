import json

import cv2
import numpy as np
import pandas as pd
import pytest

from rabbitsense import zeroshot


def test_crop_is_centered_square_and_pads_outside_image():
    image = np.zeros((100, 200, 3), np.uint8)
    image[40:60, 90:110] = 255
    patch = zeroshot.crop(image, [90, 40, 20, 20], size=40)
    assert patch.shape == (40, 40, 3)
    assert patch[10:30, 10:30].min() == 255 and patch[:8].max() == 0
    assert zeroshot.crop(image, [190, 90, 20, 20], size=40).shape == (40, 40, 3)


def test_band_window_and_missing_guides():
    crops = {k: np.full((4, 4, 3), k, np.uint8) for k in range(17)}
    crops[14] = None
    seen = []

    def scorer(clips):
        seen.extend(int(c[0, 0, 0, 0]) for c in clips)
        return [.9, .5, .1][:len(clips)] + [.9] * max(0, len(clips) - 3)

    rows = zeroshot.classify(crops, 4., 17, scorer)
    assert [r['seconds'] for r in rows] == [0., .5, 1., 1.5, 2., 2.5, 3., 3.5, 4.]
    assert [r['reason'] for r in rows[:4]] == ['first_sample', 'window_outside_video', 'window_outside_video',
                                               'zero_shot_locomotion']
    assert seen == [1, 3, 5]
    assert [r['movement'] for r in rows[3:6]] == ['locomotion_candidate', 'unknown', 'low_motion_candidate']
    assert rows[6]['reason'] == 'guide_missing'
    assert rows[-1]['reason'] == 'window_outside_video'


def test_evaluate_end_to_end_with_labels(tmp_path):
    writer = cv2.VideoWriter(str(tmp_path / 'IMG_1.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 8., (64, 48))
    for frame in range(65):
        image = np.full((48, 64, 3), 40, np.uint8)
        image[10:30, 10:30] = 200 if frame < 32 else 90
        writer.write(image)
    writer.release()
    (tmp_path / 'a.json').write_text(json.dumps(dict(width=64, height=48, marks=[dict(time=0., box=[10, 10, 20, 20]),
                                                                              dict(time=8., box=[10, 10, 20, 20])])))
    (tmp_path / 'labels.csv').write_text('clip_id,video_id,reviewed,label,start_seconds,duration_seconds\n'
                                         'a,IMG_1.avi,si,locomotion,0,4\nb,IMG_1.avi,si,resting,4,4\n')
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=[dict(name='IMG_1', video='IMG_1.avi', anchors='a.json',
                                                                 labels='labels.csv')])))
    scorer = lambda clips: [.9 if c[:, 112, 112].mean() > 150 else .1 for c in clips]
    result = zeroshot.evaluate(tmp_path / 'm.json', tmp_path / 'out', scorer)
    xclip = result['methods']['xclip']
    assert (xclip['locomotion_hits'], xclip['resting_correct'], xclip['resting_false_events']) == (1, 1, 0)
    assert not xclip['passed']
    frame = pd.read_csv(tmp_path / 'out' / 'IMG_1' / 'motion.csv')
    assert frame.seconds.iloc[-1] == 8. and frame.movement.iloc[-1] == 'unknown'
    assert json.loads((tmp_path / 'out' / 'verdict.json').read_text())['method']['band'] == [.4, .6]


def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def test_xclip_scorer_runs_real_model_code_on_tiny_random_weights(tmp_path):
    for module in ('torch', 'torchvision', 'PIL'):
        pytest.importorskip(module)
    transformers = pytest.importorskip('transformers')
    chars = list(bytes_to_unicode().values())
    vocab = {t: i for i, t in enumerate(chars + [c + '</w>' for c in chars] + ['<|startoftext|>', '<|endoftext|>'])}
    config = transformers.XCLIPConfig(
        text_config=dict(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, max_position_embeddings=77),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                           image_size=224, patch_size=32, num_frames=8, mit_hidden_size=32, mit_intermediate_size=64,
                           mit_num_hidden_layers=1, mit_num_attention_heads=2),
        projection_dim=32, prompt_layers=1, prompt_attention_heads=2)
    transformers.XCLIPModel(config).save_pretrained(tmp_path)
    transformers.XCLIPProcessor(image_processor=transformers.VideoMAEImageProcessor(),
                                tokenizer=transformers.CLIPTokenizer(vocab=vocab, merges=[])).save_pretrained(tmp_path)
    scorer = zeroshot.XClipScorer(str(tmp_path), 'cpu')
    p = scorer([np.random.default_rng(i).integers(0, 255, (8, 224, 224, 3), dtype=np.uint8) for i in range(3)])
    assert p.shape == (3,) and np.all((p > 0) & (p < 1))
    assert scorer.info['transformers'] == transformers.__version__
