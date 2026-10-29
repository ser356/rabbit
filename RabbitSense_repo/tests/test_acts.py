import json
import os

import cv2
import numpy as np
import pandas as pd
import pytest

from rabbitsense import acts


def test_ethogram_keys_codes_and_priority_are_fixed():
    assert len({a['key'] for a in acts.ETHOGRAM}) == len(acts.ETHOGRAM) == 9
    assert acts.CODES == ['binky', 'rearing', 'locomotion', 'eating', 'grooming', 'exploring', 'lying', 'loaf',
                          'not_visible']
    assert all(a['definition'] for a in acts.ETHOGRAM)


def test_census_counts_bouts_windows_sessions_and_evaluability(tmp_path):
    rows = []
    for v, name in enumerate(['5177', 'IMG_5190', 'IMG_5191', 'WhatsApp a', 'WhatsApp b', 'resting']):
        rows += [dict(video=name, start_seconds=0., end_seconds=5., act='eating', finished=1),
                 dict(video=name, start_seconds=5., end_seconds=6.5, act='grooming', finished=1),
                 dict(video=name, start_seconds=6.5, end_seconds=8., act='not_visible', finished=1)]
    pd.DataFrame(rows).to_csv(tmp_path / 'a.csv', index=False)
    report = acts.census(tmp_path / 'a.csv', tmp_path / 'censo.json')
    by = {r['act']: r for r in report['acts']}
    assert by['eating']['bouts'] == 6 and by['eating']['disjoint_windows'] == 12 and by['eating']['videos'] == 6
    assert by['eating']['sessions'] == 4  # IMG_5190/5191 y los WhatsApp cuentan como una sesión cada uno
    assert by['eating']['evaluable'] is False  # 12 ventanas < 20
    assert by['grooming']['disjoint_windows'] == 0 and by['binky']['seconds'] == 0
    assert report['finished_videos'] == 6 and (tmp_path / 'censo.json').is_file()


def test_load_rejects_unknown_acts_and_overlaps(tmp_path):
    pd.DataFrame([dict(video='a', start_seconds=0., end_seconds=2., act='dancing')]).to_csv(tmp_path / 'x.csv', index=False)
    with pytest.raises(ValueError, match='desconocidos'):
        acts.load(tmp_path / 'x.csv')
    pd.DataFrame([dict(video='a', start_seconds=0., end_seconds=2., act='eating'),
                  dict(video='a', start_seconds=1., end_seconds=3., act='loaf')]).to_csv(tmp_path / 'y.csv', index=False)
    with pytest.raises(ValueError, match='solapados'):
        acts.load(tmp_path / 'y.csv')


def test_page_marks_acts_with_keys_and_exports_segments(tmp_path):
    playwright = pytest.importorskip('playwright.sync_api')
    video = tmp_path / 'clip.webm'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'VP80'), 10., (64, 48))
    for i in range(60):
        writer.write(np.full((48, 64, 3), i * 4, np.uint8))
    writer.release()
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=[dict(name='clip', video='clip.webm', labels='x.csv')])))
    page = acts.page(tmp_path / 'm.json', tmp_path / 'out')
    html = page.read_text()
    assert 'Masticar' not in html and 'Mastica o arranca comida' in html
    try:
        with playwright.sync_playwright() as p:
            executable = os.environ.get('CHROMIUM_PATH')
            browser = p.chromium.launch(executable_path=executable) if executable else p.chromium.launch()
            tab = browser.new_page(accept_downloads=True)
            tab.goto(page.as_uri())
            tab.wait_for_function('document.getElementById("v").readyState >= 1', timeout=5000)
            tab.evaluate('document.getElementById("v").currentTime = 0')
            tab.keyboard.press('5')
            tab.evaluate('document.getElementById("v").currentTime = 2.5')
            tab.wait_for_timeout(200)
            tab.keyboard.press('2')
            tab.evaluate('document.getElementById("v").currentTime = 4')
            tab.wait_for_timeout(200)
            tab.keyboard.press('3')
            tab.keyboard.press('Backspace')  # deshace la última marca
            tab.click('#done')
            with tab.expect_download() as download:
                tab.click('#save')
            path = tmp_path / 'actos.csv'
            download.value.save_as(path)
            browser.close()
    except Exception as error:  # navegador no disponible
        pytest.skip(str(error))
    table = acts.load(path)
    assert list(table.act) == ['loaf', 'eating']
    assert table.start_seconds.tolist() == pytest.approx([0., 2.5]) and table.end_seconds.iloc[-1] == pytest.approx(6., abs=.2)
    assert table.finished.tolist() == [1, 1]


def test_sheets_crop_around_the_mask_in_real_time_and_segments_merge(tmp_path):
    video = tmp_path / 'v.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10., (320, 180))
    (tmp_path / 'm' / 'masks').mkdir(parents=True)
    for f in range(100):
        image = np.full((180, 320, 3), 50, np.uint8)
        image[80:100, 20 + 2 * f:50 + 2 * f] = 255
        writer.write(image)
        mask = np.zeros((180, 320), np.uint8)
        if f < 90:
            mask[80:100, 20 + 2 * f:50 + 2 * f] = 255
        cv2.imwrite(str(tmp_path / 'm' / 'masks' / f'{f:06d}.png'), mask)
    writer.release()
    (tmp_path / 'man.json').write_text(json.dumps(dict(cases=[dict(name='v', video='v.mp4', masks='m', labels='x')])))
    count = acts.sheets(tmp_path / 'man.json', tmp_path / 'sheets')
    frames = pd.read_csv(tmp_path / 'sheets' / 'fotogramas.csv')
    assert count == len(frames) == 20 and frames.seconds.iloc[1] == .5 and frames.frame.iloc[1] == 5
    assert frames.masked.tolist() == [True] * 18 + [False] * 2
    sheet = cv2.imread(str(tmp_path / 'sheets' / 'v_00.jpg'))
    assert sheet.shape == (3 * 200, 70 + 8 * 200, 3)
    first = sheet[:200, 70:270]
    assert first[90:110, 80:120].mean() > 150  # el objeto claro queda centrado en el recorte
    labels = frames[['case', 'seconds']].assign(act=['loaf'] * 6 + ['eating'] * 10 + ['not_visible'] * 4)
    labels.to_csv(tmp_path / 'l.csv', index=False)
    table = acts.segments(tmp_path / 'l.csv', tmp_path / 'sheets' / 'fotogramas.csv', tmp_path / 's.csv')
    assert table.act.tolist() == ['loaf', 'eating', 'not_visible']
    assert table.start_seconds.tolist() == [0., 3., 8.] and table.end_seconds.tolist() == [3., 8., 10.]
    labels.iloc[:-1].to_csv(tmp_path / 'short.csv', index=False)
    with pytest.raises(ValueError, match='Faltan 1'):
        acts.segments(tmp_path / 'short.csv', tmp_path / 'sheets' / 'fotogramas.csv', tmp_path / 's2.csv')


def test_verification_page_is_blind_stratified_and_agreement_is_computed(tmp_path):
    playwright = pytest.importorskip('playwright.sync_api')
    video = tmp_path / 'clip.webm'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'VP80'), 10., (64, 48))
    for i in range(200):
        writer.write(np.full((48, 64, 3), i, np.uint8))
    writer.release()
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=[dict(name='clip', video='clip.webm', labels='x')])))
    pd.DataFrame([dict(video='clip', start_seconds=0., end_seconds=8., act='loaf', finished=1),
                  dict(video='clip', start_seconds=8., end_seconds=13., act='eating', finished=1),
                  dict(video='clip', start_seconds=13., end_seconds=14., act='grooming', finished=1),
                  dict(video='clip', start_seconds=14., end_seconds=20., act='not_visible', finished=1)]
                 ).to_csv(tmp_path / 'l.csv', index=False)
    page = acts.verify_page(tmp_path / 'm.json', tmp_path / 'l.csv', tmp_path / 'out', dict(acts.VERIFY, fragments=6))
    mapping = pd.read_csv(tmp_path / 'out' / 'verificacion_mapa.csv')
    assert len(mapping) == 6 and set(mapping.act) == {'loaf', 'eating', 'not_visible'}  # grooming: segmento < 2 s
    assert (mapping.end - mapping.start).to_numpy() == pytest.approx(np.full(6, 2.))
    html = page.read_text()
    items = json.loads(html.split('const items = ')[1].split(', acts =')[0])
    assert all(set(x) == {'code', 'src', 'start', 'end'} for x in items)
    try:
        with playwright.sync_playwright() as p:
            executable = os.environ.get('CHROMIUM_PATH')
            browser = p.chromium.launch(executable_path=executable) if executable else p.chromium.launch()
            tab = browser.new_page(accept_downloads=True)
            tab.goto(page.as_uri())
            for code in mapping.act:
                tab.keyboard.press(next(a['key'] for a in acts.ETHOGRAM if a['code'] == code))
            with tab.expect_download() as download:
                tab.click('#save')
            download.value.save_as(tmp_path / 'r.csv')
            browser.close()
    except Exception as error:
        pytest.skip(str(error))
    answers = pd.read_csv(tmp_path / 'r.csv')
    answers.loc[0, 'act'] = 'exploring'
    answers.to_csv(tmp_path / 'r.csv', index=False)
    report = acts.agreement(tmp_path / 'r.csv', tmp_path / 'out' / 'verificacion_mapa.csv', tmp_path / 'a.json')
    assert report['fragments'] == 6 and report['agreement'] == pytest.approx(5 / 6) and report['kappa'] < 1


def test_motion_sheets_light_up_chewing_but_not_a_still_body(tmp_path):
    """Cuerpo quieto con 'boca' que oscila a 4 Hz (vídeo 1280 px, máscaras a resolución original)."""
    fps, size = 30., (1280, 720)
    video = tmp_path / 'chew.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    (tmp_path / 'm' / 'masks').mkdir(parents=True)
    rng = np.random.default_rng(0)
    floor = rng.integers(60, 120, (720, 1280, 3)).astype(np.uint8)
    for f in range(120):
        image = floor.copy()
        image[300:420, 500:700] = 200  # cuerpo
        if f < 60:  # primera ventana: mastica
            dy = int(6 * np.sin(2 * np.pi * 4 * f / fps))
            image[400 + dy:420 + dy, 700:740] = 40  # boca
        else:
            image[400:420, 700:740] = 40
        writer.write(image)
        mask = np.zeros((720, 1280), np.uint8)
        mask[300:430, 500:740] = 255
        cv2.imwrite(str(tmp_path / 'm' / 'masks' / f'{f:06d}.png'), mask)
    writer.release()
    (tmp_path / 'man.json').write_text(json.dumps(dict(cases=[dict(name='chew', video='chew.mp4', masks='m', labels='x')])))
    assert acts.motion_sheets(tmp_path / 'man.json', tmp_path / 'out') == 2
    windows = pd.read_csv(tmp_path / 'out' / 'ventanas.csv')
    assert windows.masked.all() and windows.frames.tolist() == [60, 60]
    assert windows.motion_p999[0] > 5 * max(windows.motion_p999[1], .5)
    sheet = cv2.imread(str(tmp_path / 'out' / 'chew_00.jpg'))
    assert sheet.shape == (2 * 200, 70 + 5 * 200, 3)
    heat = sheet[:200, 70 + 4 * 200:]
    assert heat.max() > 150 and sheet[200:, 70 + 4 * 200:].mean() < heat.mean()
    labels = pd.DataFrame(dict(case=['chew', 'chew'], start=[0., 2.], act=['eating', 'loaf']))
    labels.to_csv(tmp_path / 'w.csv', index=False)
    table = acts.window_segments(tmp_path / 'w.csv', tmp_path / 'seg.csv')
    assert table.act.tolist() == ['eating', 'loaf'] and table.end_seconds.tolist() == [2., 4.]


def test_second_verification_avoids_previous_fragments_and_storage(tmp_path):
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=[dict(name='clip', video='clip.webm', labels='x')])))
    pd.DataFrame([dict(video='clip', start_seconds=0., end_seconds=20., act='loaf', finished=1),
                  dict(video='clip', start_seconds=20., end_seconds=40., act='eating', finished=1)]
                 ).to_csv(tmp_path / 'l.csv', index=False)
    acts.verify_page(tmp_path / 'm.json', tmp_path / 'l.csv', tmp_path / 'a', dict(acts.VERIFY, fragments=6))
    page = acts.verify_page(tmp_path / 'm.json', tmp_path / 'l.csv', tmp_path / 'b', dict(acts.VERIFY, fragments=6, seed=1),
                            [tmp_path / 'a' / 'verificacion_mapa.csv'])
    old, new = (pd.read_csv(tmp_path / d / 'verificacion_mapa.csv') for d in 'ab')
    assert len(new) == 6 and not set(old.code) & set(new.code)
    assert not any(a.start < b.end and a.end > b.start for a in new.itertuples() for b in old.itertuples())
    assert not any(a.start < b.end and a.end > b.start for a in new.itertuples() for b in new.itertuples() if a.code != b.code)
    assert "rabbitsense_verificacion_actos_1_" in page.read_text()
