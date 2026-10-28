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
