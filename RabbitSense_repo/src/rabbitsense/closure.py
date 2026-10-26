import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .evaluate import clip_metrics, run_suite
from .motion import sha256, timeline

# Fixed before running on the videos; changing them after seeing results invalidates the verdict.
CRITERIA = {'minimum_locomotion_clips': 5, 'minimum_resting_clips': 10,
            'sensitivity': .8, 'specificity': .8, 'hit_per_locomotion_video': 1}


def tally(cases):
    """cases: {name: clip_metrics result}. Pools clip-level detection over videos."""
    result = dict(locomotion_hits=0, locomotion_clips=0, resting_correct=0, resting_clips=0,
                  resting_false_events=0, per_video={})
    for name, clips in cases.items():
        detection = clips['detection']
        result['per_video'][name] = detection
        for key in ('locomotion_hits', 'locomotion_clips', 'resting_correct', 'resting_clips', 'resting_false_events'):
            result[key] += detection[key]
    return result


def judge(counts, criteria=CRITERIA):
    sensitivity = counts['locomotion_hits'] / counts['locomotion_clips'] if counts['locomotion_clips'] else 0.
    specificity = counts['resting_correct'] / counts['resting_clips'] if counts['resting_clips'] else 0.
    checks = {
        'enough_locomotion_clips': counts['locomotion_clips'] >= criteria['minimum_locomotion_clips'],
        'enough_resting_clips': counts['resting_clips'] >= criteria['minimum_resting_clips'],
        'sensitivity': sensitivity >= criteria['sensitivity'],
        'specificity': specificity >= criteria['specificity'],
        'hit_in_every_locomotion_video': all(d['locomotion_hits'] >= criteria['hit_per_locomotion_video']
                                             for d in counts['per_video'].values() if d['locomotion_clips']),
    }
    return dict(passed=all(checks.values()), checks=checks, sensitivity=sensitivity, specificity=specificity, **counts)


def verdict(suite, segmentation=None, criteria=CRITERIA):
    methods = {method: {name: report['motion'][method]['legacy_clips'] for name, report in suite.items()}
               for method in next(iter(suite.values()))['motion']}
    if segmentation is not None:
        methods['sam2_masked_flow'] = {name: case['metrics']['masked_flow']['clips']
                                       for name, case in segmentation['cases'].items()}
    results = {method: judge(tally(cases), criteria) for method, cases in methods.items()}
    passing = sorted(m for m, r in results.items() if r['passed'])
    status = ('closed_resolved_on_development_benchmark' if passing else
              'closed_not_resolvable_with_existing_data')
    return dict(status=status, passing_methods=passing, criteria=criteria, methods=results,
                scope='Reviewed legacy clip labels on development videos; not frame-level truth, not an independent '
                      'test and not a welfare indicator. A pass means candidate locomotion episodes appear in '
                      'locomotion clips and measured low motion without episodes in resting clips.')


def evaluate_cases(manifest, out, method, analyze, info):
    """Ejecuta `analyze(paths) -> filas` por caso del manifest y aplica el mismo veredicto que el resto de métodos."""
    manifest, out = Path(manifest).resolve(), Path(out)
    suite, inputs = {}, {}
    for case in json.loads(manifest.read_text())['cases']:
        paths = {key: (manifest.parent / Path(case[key]).expanduser()).resolve()
                 for key in ('video', 'anchors', 'labels', 'masks') if key in case}
        destination = out / case['name']
        destination.mkdir(parents=True, exist_ok=True)
        rows = analyze(paths)
        frame = pd.DataFrame(rows)
        frame.to_csv(destination / 'motion.csv', index=False)
        (destination / 'timeline.json').write_text(json.dumps(timeline(rows), indent=2))
        clips = clip_metrics(frame, pd.read_csv(paths['labels']), paths['video'].name)
        suite[case['name']] = dict(motion={method: dict(legacy_clips=clips)})
        inputs[case['name']] = {key: dict(path=str(path), sha256=sha256(path))
                                for key, path in paths.items() if path.is_file()}
        print(case['name'], frame.movement.value_counts().to_dict(), flush=True)
    result = verdict(suite)
    result.update(inputs=inputs, manifest_sha256=sha256(manifest), method=info,
                  source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                  clips={name: case['motion'][method]['legacy_clips'] for name, case in suite.items()})
    out.mkdir(parents=True, exist_ok=True)
    (out / 'verdict.json').write_text(json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False))
    return result


def copy_masks(source, destination):
    source, destination = Path(source).resolve(strict=True), Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copytree(source / 'masks', destination / 'masks')
    for name in ('summary.json', 'tracks.csv'):
        shutil.copy2(source / name, destination / name)
    return destination


def close(suite_path, out, segmentation_sources=None, announce=True):
    out = Path(out)
    suite = run_suite(suite_path, out / 'suite')
    segmentation = None
    if segmentation_sources:
        from .segment import assess_masks, measure_masks
        missing = set(suite) - set(segmentation_sources)
        if missing:
            raise ValueError(f'Faltan máscaras SAM2 para: {sorted(missing)}')
        directories = {name: copy_masks(path, out / 'segmentation' / name) for name, path in segmentation_sources.items()}
        for directory in directories.values():
            measure_masks(directory)
        segmentation = assess_masks(suite_path, out / 'suite' / 'suite.json', directories,
                                    out / 'segmentation' / 'comparison.json')
    result = verdict(suite, segmentation)
    result['inputs'] = {'manifest_sha256': sha256(suite_path),
                        'source_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    (out / 'verdict.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"\n{'Método':<18} {'Sens.':>7} {'Espec.':>7} {'Falsos':>7}  Resultado")
    for method, r in result['methods'].items():
        print(f"{method:<18} {r['locomotion_hits']}/{r['locomotion_clips']:<5} "
              f"{r['resting_correct']}/{r['resting_clips']:<5} {r['resting_false_events']:>7}  "
              f"{'PASA' if r['passed'] else 'NO PASA'}")
    if announce:
        print(f"\nEstado: {result['status']}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Cierre de locomoción con criterios fijados antes de ejecutar')
    parser.add_argument('--suite', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--segmentation', action='append', default=[], metavar='CASO=DIR',
                        help='Máscaras SAM2 existentes; se copian, nunca se sobrescriben. Todos los casos o ninguno.')
    args = parser.parse_args()
    close(args.suite, args.out, dict(item.split('=', 1) for item in args.segmentation))


if __name__ == '__main__':
    main()
