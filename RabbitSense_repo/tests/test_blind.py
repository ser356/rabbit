import json
import os

import cv2
import numpy as np
import pandas as pd
import pytest

from rabbitsense import blind
from rabbitsense.motion import sha256

REPO_MANIFEST = blind.REPO / 'benchmarks' / 'pose_desarrollo.json'


def test_page_covers_every_verified_clip_and_hides_labels(tmp_path):
    page = blind.page(REPO_MANIFEST, tmp_path)
    mapping = pd.read_csv(tmp_path / 'ciego_mapa.csv')
    assert len(mapping) == 57 and mapping.code.is_unique and mapping.clip_id.is_unique
    html = page.read_text()
    clips = json.loads(html.split('const clips = ')[1].split(';\n')[0])
    assert [c['code'] for c in clips] == list(mapping.code)
    assert set(clips[0]) == {'code', 'src', 'start', 'end'}
    for leak in ('mano_IMG', 'v24_', 'Verificado', 'propuesta previa'):
        assert leak not in html
    assert 'locomotion' not in json.dumps(clips) and 'resting' not in json.dumps(clips)
    order = list(mapping.clip_id)
    assert order != sorted(order) and order == list(pd.read_csv(tmp_path / 'ciego_mapa.csv').clip_id)
    blind.page(REPO_MANIFEST, tmp_path / 'again')
    assert list(pd.read_csv(tmp_path / 'again' / 'ciego_mapa.csv').code) == list(mapping.code)  # misma semilla


def test_apply_writes_labels_manifest_and_agreement(tmp_path):
    blind.page(REPO_MANIFEST, tmp_path)
    mapping = pd.read_csv(tmp_path / 'ciego_mapa.csv')
    old = blind.verified_clips(REPO_MANIFEST).set_index('clip_id').old_label
    answers = mapping[['code']].assign(label=[old[c] for c in mapping.clip_id])
    answers.loc[0, 'label'] = 'unknown'
    answers.to_csv(tmp_path / 'r.csv', index=False)
    with pytest.raises(ValueError, match='Faltan'):
        answers.iloc[1:].to_csv(tmp_path / 'short.csv', index=False)
        blind.apply(REPO_MANIFEST, tmp_path / 'short.csv', tmp_path / 'ciego_mapa.csv', tmp_path / 'l.csv',
                    tmp_path / 'm.json', tmp_path / 'rep.json')
    report = blind.apply(REPO_MANIFEST, tmp_path / 'r.csv', tmp_path / 'ciego_mapa.csv', tmp_path / 'l.csv',
                         tmp_path / 'm.json', tmp_path / 'rep.json')
    assert report['clips'] == 57 and report['agreement'] == pytest.approx(56 / 57) and len(report['changed']) == 1
    assert 0.9 < report['kappa_old_vs_new'] < 1 and report['answers_sha256'] == sha256(tmp_path / 'r.csv')
    labels = pd.read_csv(tmp_path / 'l.csv')
    assert len(labels) == 57 and (labels.reviewed == 'si').all() and set(labels.columns) >= {
        'clip_id', 'video_id', 'start_seconds', 'duration_seconds', 'label'}
    spec = json.loads((tmp_path / 'm.json').read_text())
    original = json.loads(REPO_MANIFEST.read_text())
    assert [c['name'] for c in spec['cases']] == [c['name'] for c in original['cases']]
    for new, old_case in zip(spec['cases'], original['cases']):
        assert (tmp_path / new['labels']).resolve() == (tmp_path / 'l.csv').resolve()
        for key in ('video', 'masks'):
            if old_case[key].startswith(('~', '/')):
                assert new[key] == old_case[key]
            else:
                assert (tmp_path / new[key]).resolve() == (REPO_MANIFEST.parent / old_case[key]).resolve()
    rebuilt = blind.verified_clips(tmp_path / 'm.json')
    assert len(rebuilt) == 56  # el clip pasado a desconocido no se evalúa


def test_kappa():
    assert blind.kappa(['a', 'b', 'a', 'b'], ['a', 'b', 'a', 'b']) == 1.
    assert blind.kappa(['a', 'a', 'b', 'b'], ['a', 'b', 'a', 'b']) == 0.


def test_page_records_keys_and_exports_csv(tmp_path):
    playwright = pytest.importorskip('playwright.sync_api')
    video = tmp_path / 'v.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10., (64, 48))
    for _ in range(80):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    labels = pd.DataFrame([dict(clip_id=f'c{k}', video_id='v.mp4', start_seconds=4. * k, duration_seconds=4.,
                                label='locomotion' if k else 'resting', reviewed='si') for k in range(2)])
    labels.to_csv(tmp_path / 'labels.csv', index=False)
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=[dict(name='v', video='v.mp4', labels='labels.csv')])))
    page = blind.page(tmp_path / 'm.json', tmp_path / 'out')
    try:
        with playwright.sync_playwright() as p:
            executable = os.environ.get('CHROMIUM_PATH')
            browser = p.chromium.launch(executable_path=executable) if executable else p.chromium.launch()
            tab = browser.new_page(accept_downloads=True)
            tab.goto(page.as_uri())
            tab.keyboard.press('l')
            tab.keyboard.press('u')
            assert 'respondidos 2' in tab.inner_text('#status')
            with tab.expect_download() as download:
                tab.click('#save')
            path = tmp_path / 'respuestas.csv'
            download.value.save_as(path)
            browser.close()
    except Exception as error:  # navegador no disponible en este entorno
        pytest.skip(str(error))
    answers = pd.read_csv(path)
    mapping = pd.read_csv(tmp_path / 'out' / 'ciego_mapa.csv')
    assert list(answers.code) == list(mapping.code) and list(answers.label) == ['locomotion', 'unknown']
