"""Independent post-generation table audit; expected CSVs are validation-only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

KEYS = {
    'model_summary': ('model',), 'accuracy_axes': ('model', 'axis', 'subset'),
    'cell_metrics': ('model', 'stratum', 'image_op', 'text_op', 'image_strength', 'text_strength'),
    'pair_metrics': ('model', 'stratum', 'image_op', 'text_op'), 'root_metrics': ('model', 'root'),
    'source_metrics': ('model', 'source'), 'slice_coverage': ('model', 'slice'),
    'missed_locations': ('model', 'slice', 'image_strength', 'text_strength'),
    'recurrence': ('model', 'threshold'), 'budget_curves': ('model', 'policy', 'budget'),
    'model_replacement': ('old_model', 'new_model', 'cohort'), 'source_equal_coverage': ('model',),
}
CATEGORICAL = {'model', 'axis', 'subset', 'stratum', 'image_op', 'text_op', 'root', 'image_group',
               'source', 'slice', 'policy', 'old_model', 'new_model', 'cohort', 'interval'}
BOOLEAN = {'clean_pass', 'strict41', 'full_suite_pass', 'diagonal_failure', 'corner_failure'}
FLOATS = {'accuracy', 'binary_retention', 'affected_fraction', 'strict41_event_density',
          'median_events_affected', 'all_exact_gap_rate', 'eligible_rate', 'gap_rate', 'gap_total_rate',
          'task_accuracy', 'semantic_accuracy', 'format_rate', 'clean_task_accuracy', 'image_task_accuracy',
          'text_task_accuracy', 'normalized_product_excess', 'clean_accuracy', 'single_accuracy', 'joint_accuracy',
          'suite_failure_rate', 'discovery_fraction', 'preservation_rate', 'source_equal_coverage'}


def read(path, keys):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(set(fields)) != len(fields) or not set(keys) <= set(fields):
            raise AssertionError(f'Bad/duplicate header: {path}')
        rows = {}
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise AssertionError(f'Incomplete row: {path}')
            key = tuple(row[k] for k in keys)
            if key in rows:
                raise AssertionError(f'Duplicate key {key}: {path}')
            rows[key] = row
        return fields, rows


def compare(actual_dir, expected_dir):
    actual_dir, expected_dir = Path(actual_dir), Path(expected_dir)
    observed = {p.stem for p in actual_dir.glob('*.csv')}
    if observed != set(KEYS):
        raise AssertionError(f'Table set mismatch: {observed ^ set(KEYS)}')
    report = {}
    for name, keys in KEYS.items():
        actual = actual_dir / f'{name}.csv'
        expected = expected_dir / f'{name}.csv'
        af, ar = read(actual, keys)
        ef, er = read(expected, keys)
        if af != ef:
            raise AssertionError(f'{name}: ordered header mismatch: {af} vs {ef}')
        if set(ar) != set(er):
            raise AssertionError(f'{name}: row key mismatch')
        counts = dict(categorical=0, integer=0, boolean=0, floating=0, null=0)
        max_delta = 0.0
        for key, row in ar.items():
            for column in af:
                a, e = row[column], er[key][column]
                where = f'{name} {key} {column}: {a!r} vs {e!r}'
                if a == '' or e == '':
                    if a != e:
                        raise AssertionError(where)
                    counts['null'] += 1
                elif column in CATEGORICAL:
                    if a != e:
                        raise AssertionError(where)
                    counts['categorical'] += 1
                elif column in BOOLEAN:
                    if a not in ('True', 'False') or a != e:
                        raise AssertionError(where)
                    counts['boolean'] += 1
                elif column in FLOATS or (column == 'discovered_roots' and row.get('policy') == 'uniform_expected'):
                    av, ev = float(a), float(e)
                    if not math.isfinite(av) or not math.isfinite(ev) or abs(av-ev) > 1e-9:
                        raise AssertionError(where)
                    max_delta = max(max_delta, abs(av-ev))
                    counts['floating'] += 1
                else:
                    if not re.fullmatch(r'-?\d+', a) or not re.fullmatch(r'-?\d+', e) or int(a) != int(e):
                        raise AssertionError(where)
                    counts['integer'] += 1
        report[name] = dict(status='PASS', rows=len(ar), columns=af, key_columns=keys, compared_values=counts,
            max_float_absolute_delta=max_delta, float_absolute_tolerance=1e-9,
            actual_sha256=hashlib.sha256(actual.read_bytes()).hexdigest(), expected_sha256=hashlib.sha256(expected.read_bytes()).hexdigest(),
            row_column_key_completeness=True)
    return dict(status='PASS', table_count=len(report), tables=report, expected_csvs_used_only_after_generation=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--expected', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.actual, args.expected)
    with args.report.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], tables=result['table_count'], rows=sum(v['rows'] for v in result['tables'].values()),
                          maximum_float_delta=max(v['max_float_absolute_delta'] for v in result['tables'].values()))))
