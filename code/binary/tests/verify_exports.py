"""Post-generation scientific comparator; never imported by either producer.

Compare every ordered header, row, field and shared draw to frozen references.
Integer/categorical/null values must agree exactly; finite floats use 1e-12.
Receipts, timings and absolute paths are deliberately not scientific comparands.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import numpy as np


def compare_csv(actual, expected):
    with actual.open(encoding='utf-8-sig',newline='') as a, expected.open(encoding='utf-8-sig',newline='') as e:
        aa, ee = list(csv.reader(a)), list(csv.reader(e))
    assert aa[0] == ee[0], f'{actual.name}: header mismatch'
    assert len(aa) == len(ee), f'{actual.name}: row count mismatch'
    maximum, values = 0., 0
    for i, (ar, er) in enumerate(zip(aa[1:],ee[1:]), 1):
        assert len(ar)==len(er)==len(aa[0]), f'{actual.name}:{i}: ragged row'
        for column, av, ev in zip(aa[0],ar,er):
            values += 1
            if av == ev:
                continue
            where=f'{actual.name}:{i}:{column}: {av!r} vs {ev!r}'
            if re.fullmatch(r'-?\d+',ev) or ev in ('','True','False'):
                raise AssertionError(where)
            try:
                af,ef=float(av),float(ev)
            except ValueError:
                raise AssertionError(where) from None
            assert math.isfinite(af) and math.isfinite(ef) and abs(af-ef)<=1e-12, where
            maximum=max(maximum,abs(af-ef))
    return dict(rows=len(aa)-1,fields=len(aa[0]),scalar_checks=values,max_absolute_float_error=maximum)


def verify(actual, legacy, metrics):
    actual,legacy,metrics=map(Path,(actual,legacy,metrics))
    report={}
    for name,expected,count in (('binary_v1',legacy,12),('joint_metrics_v1',metrics,18)):
        files=sorted((actual/name).glob('*.csv'))
        assert len(files)==count, f'{name}: expected {count} CSVs'
        if name=='joint_metrics_v1':
            assert {p.name for p in files}=={p.name for p in expected.glob('*.csv')}
        report[name]={p.name:compare_csv(p,expected/p.name) for p in files}
    with np.load(actual/'joint_metrics_v1'/'bootstrap_draws.npz',allow_pickle=False) as a, np.load(metrics/'bootstrap_draws.npz',allow_pickle=False) as e:
        assert set(a.files)==set(e.files)
        for key in a.files:
            assert np.array_equal(a[key],e[key]), f'bootstrap member {key} differs'
        draws_shape=list(a['draws'].shape)
    result=dict(status='PASS',float_absolute_tolerance=1e-12,tables=report,
                csv_count=30,shared_draw_matrix_equal=True,draws_shape=draws_shape,
                expected_tables_used_only_after_generation=True,
                excluded_non_scientific_fields=['timings','absolute paths','run identifiers'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual',type=Path,required=True)
    parser.add_argument('--legacy',type=Path,required=True)
    parser.add_argument('--metrics',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    result=verify(args.actual,args.legacy,args.metrics)
    with args.report.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','csv_count','shared_draw_matrix_equal','draws_shape')}))
