"""Replay frozen public labels into twelve legacy and eighteen paired-metric CSVs.

CPU/NumPy only unless --plots is selected. This does not run models or grading.
Every output directory must be new; partial results and failures are retained.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

from reproduce_binary import PINS, TABLE_NAMES, pin, reproduce
from joint_metrics import PROTOCOL_SHA256, THREAD_VARS, run as metrics_run

CODE = Path(__file__).resolve().parent
RELEASE = CODE.parents[1]
METRIC_TABLES = ('summary', 'cells', 'tail_sensitivity', 'H_curve', 'H_thresholds',
                 'source_summary', 'source_cells', 'source_equal', 'cohort_roots',
                 'pair_suites', 'pair_cells', 'pair_sources', 'matched_candidates',
                 'matched_selected', 'intervals', 'paired_contrasts',
                 'bootstrap_values', 'valid_draw_ids')


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def reproduce_all(data=None, output=None, bootstrap=2000, seed=17, plots=False, protocol=None):
    if output is None:
        raise ValueError('A fresh output path is required')
    if bootstrap < 0:
        raise ValueError('bootstrap must be nonnegative')
    # Validate before output creation: the standalone legacy producer has an
    # intentionally retained protocol-error receipt limitation.
    protocol = Path(protocol).resolve() if protocol else CODE/'frozen_protocol_v2.md'
    protocol_pin = pin(protocol)
    if protocol_pin['sha256'] != PROTOCOL_SHA256:
        raise ValueError('Frozen protocol SHA256 mismatch')
    data = Path(data).resolve() if data is not None else RELEASE/'results'/'binary_v1'
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    import numpy as np
    write_json(output/'run_manifest.json', dict(
        run_id=output.name, status='started', started_utc=datetime.now(timezone.utc).isoformat(),
        sources={name: pin(CODE/name) for name in ('reproduce_all.py','reproduce_binary.py','joint_metrics.py','plot_joint_metrics.py','validate_legacy.py')},
        protocol=protocol_pin, expected_inputs=PINS, data=str(data),
        environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, platform=platform.platform(),
                         thread_pins={v: os.environ.get(v) for v in THREAD_VARS}),
        bootstrap=bootstrap, seed=seed, plots=bool(plots),
        data_role='existing public known benchmark; original root/image identities unchanged',
        model_revisions='retained released labels; see inference_contract.json; no models loaded',
        model_calls=0, judge_calls=0, new_input_exposures=0, training=False))
    try:
        legacy = reproduce(data, output/'binary_v1', protocol=protocol)
        if set(legacy['tables']) != {name+'.csv' for name in TABLE_NAMES}:
            raise ValueError('Expected exactly twelve legacy tables')
        # Reference CSVs are used only after independent mask-to-table export.
        from validate_legacy import compare
        validation = compare(output/'binary_v1', data)
        write_json(output/'legacy_validation.json', validation)
        metrics = metrics_run(data, output/'joint_metrics_v1', bootstrap=bootstrap, seed=seed, protocol=protocol)
        actual = {p.stem for p in (output/'joint_metrics_v1').glob('*.csv')}
        if actual != set(METRIC_TABLES):
            raise ValueError('Expected exactly eighteen paired metric tables')
        if plots:
            from plot_joint_metrics import plot
            plot(output/'joint_metrics_v1', output/'figures')
        files = {p.relative_to(output).as_posix(): pin(p) for p in sorted(output.rglob('*')) if p.is_file()}
        result = dict(status='completed', run_id=output.name, seconds=time.monotonic()-start,
                      legacy_tables=len(legacy['tables']), metric_tables=len(actual), files=files,
                      model_calls=0, judge_calls=0, new_input_exposures=0,
                      stopped_reason='all requested frozen-label computations and exports completed')
        write_json(output/'result.json', result)
        return result
    except Exception as exc:
        write_json(output/'failure.json', dict(status='failed', seconds=time.monotonic()-start,
                   error_type=type(exc).__name__, error=str(exc),
                   stopped_reason='exception; partial outputs preserved, no overwrite or retry'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=None, help='default: repository results/binary_v1')
    parser.add_argument('--output', type=Path, required=True, help='new parent directory; must not exist')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--protocol', type=Path, help='exact frozen v2 protocol; default bundled sibling')
    parser.add_argument('--plots', action='store_true', help='also plot exported CSVs (requires matplotlib)')
    args = parser.parse_args()
    result = reproduce_all(**vars(args))
    print(json.dumps({k: result[k] for k in ('status','run_id','seconds','legacy_tables','metric_tables')}))


if __name__ == '__main__':
    main()
