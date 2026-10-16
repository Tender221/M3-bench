"""CPU-only paired joint reliability analysis of the frozen public binary masks.

Fractions are on [0,1], undefined table values are None. JSR is qualified
joint accuracy; JSS is one minus affected-question coverage. Own cohorts are
descriptive. Common4 intervals are pointwise paired image-cluster percentiles.
"""
import argparse
import csv
from datetime import datetime, timezone
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from reproduce_binary import (ANCHORS, JOINT, PINS, load_release, pin, rate,
                              validate_masks)

PROTOCOL_SHA256 = 'f2eacb2dbbd8f5a145995c55da7875dfe13d9052e59d54a2a9d3dce2400a6996'
VIEWS = (('Task', 4, 4), ('Semantic', 1, 1), ('Semantic-on-fixed-Task', 4, 1))
METRICS = ('JSR', 'JSS', 'TGR5')
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def cell_profile(eligible, failed, q=.05, min_support=30):
    """Root-count support filtering and discrete ceil(q*V) upper-tail mean."""
    e, f = np.asarray(eligible), np.asarray(failed)
    valid = e >= min_support
    rates = f[valid] / e[valid]
    count = len(rates)
    tail = math.ceil(q * count)
    return dict(valid_cells=count, tail_cells=tail,
                TGR=float(np.sort(rates)[-tail:].mean()) if count else None,
                macro_failure=float(rates.mean()) if count else None,
                micro_failure_all=rate(int(f.sum()), int(e.sum())),
                micro_failure_valid=rate(int(f[valid].sum()), int(e[valid].sum())),
                eligible_records_valid=int(e[valid].sum()), failure_records_valid=int(f[valid].sum()))


def hotspot(rates, threshold):
    """Strict rate exceedance; threshold equality is not an exceedance."""
    return float(np.mean(np.asarray(rates) > threshold)) if len(rates) else None


def draw_images(image_count, seed, bootstrap):
    """Sorted original-image roster, PCG64, unstratified fixed-size draws."""
    return np.random.default_rng(seed).integers(0, image_count, size=(bootstrap, image_count), dtype=np.int32)


def _weights(draws, groups):
    weights = np.zeros((len(draws), groups), dtype=np.float64)
    for i, draw in enumerate(draws):
        weights[i] = np.bincount(draw, minlength=groups)
    return weights


def _divide(a, b):
    out = np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan)
    return np.divide(a, b, out=out, where=np.asarray(b) != 0)


def cluster_bootstrap(eligible_group, failed_group, strict_group, survivor_group,
                      draws, valid_cells, weights=None):
    """Aggregates carry all roots of each image; never resample qualification.

    One zero count in the original valid-cell set invalidates the entire TGR
    draw. Positive counts below the original threshold remain in its cell set.
    """
    if weights is None:
        weights = _weights(draws, len(eligible_group))
    e = weights @ np.asarray(eligible_group, dtype=float)
    f = weights @ np.asarray(failed_group, dtype=float)
    jsr = 1 - _divide(f.sum(axis=1), e.sum(axis=1))
    jss = _divide(weights @ survivor_group, weights @ strict_group)
    tgr = np.full(len(draws), np.nan)
    v = np.asarray(valid_cells, dtype=bool)
    if v.any():
        good = (e[:, v] > 0).all(axis=1)
        rates = f[good][:, v] / e[good][:, v]
        tail = math.ceil(.05 * int(v.sum()))
        tgr[good] = np.sort(rates, axis=1)[:, -tail:].mean(axis=1)
    return dict(JSR=jsr, JSS=jss, TGR5=tgr)


def _group_sum(array, group_ids, group_count):
    out = np.zeros((group_count,) + array.shape[1:], dtype=float)
    np.add.at(out, group_ids, array)
    return out


def _interval(values):
    valid = np.isfinite(values)
    low, high = np.quantile(values[valid], [.025, .975], method='linear') if valid.any() else (None, None)
    return dict(ci_low=float(low) if low is not None else None,
                ci_high=float(high) if high is not None else None,
                total_draws=len(values), valid_draws=int(valid.sum()), invalid_draws=int((~valid).sum()))


def _source_equal(rows, key):
    vals = [r[key] for r in rows]
    return float(np.mean(vals)) if vals and all(x is not None for x in vals) else None


def synthetic_examples():
    """Explicit synthetic counterexamples, unrelated to observed model outcomes."""
    eligibility = np.array([True, False])
    jsr = []
    for joint in (np.array([1, 0]), np.array([0, 1])):
        jsr.append(dict(marginals=dict(clean=.5, image_single=.5, text_single=.5, joint=float(joint.mean())),
                        eligible=eligibility.tolist(), joint=joint.tolist(),
                        JSR=float(joint[eligibility].mean())))
    jss = []
    for bad in (((0, 0), (0, 1)), ((0, 0), (1, 1))):
        joint = np.ones((2, 400), dtype=int)
        for k, c in bad:
            joint[k, c] = 0
        jss.append(dict(cell_accuracies=joint.mean(axis=0).tolist(),
                        failed_coordinates=[list(x) for x in bad], strict_roots=2,
                        joint_accuracy=float(joint.mean()), JSS=float(joint.all(axis=1).mean())))
    return dict(synthetic=True, jsr=jsr, jss=jss,
                scope='Unconditional marginals cannot identify JSR; same-cohort cell maps cannot identify JSS. Conditional cell rates with denominators do identify JSR/TGR/H.')


def analyze(flags, models, conditions, roots, bootstrap=2000, seed=17):
    """Return tables, metadata and shared draw/sample arrays; no file IO.

    The public loader enforces actual 4/1500/1065 dimensions; the pure API
    accepts small nonempty synthetic populations on the canonical 441 grid.
    """
    validate_masks(flags, models, conditions, roots)
    if len(models) != 4 or bootstrap < 0:
        raise ValueError('Expected four models and nonnegative bootstrap count')
    n = len(roots)
    image_roster = sorted({r['image_fingerprint'] for r in roots})
    sources = np.array([r['source_dataset'] for r in roots])
    source_roster = sorted(set(sources))
    image_index = {v: i for i, v in enumerate(image_roster)}
    gids = np.array([image_index[r['image_fingerprint']] for r in roots])
    g = len(image_roster)
    draws = draw_images(g, seed, bootstrap)
    weights = _weights(draws, g)
    names = ('summary', 'cells', 'tail_sensitivity', 'H_curve', 'H_thresholds',
             'source_summary', 'source_cells', 'source_equal', 'cohort_roots',
             'pair_suites', 'pair_cells', 'pair_sources', 'matched_candidates',
             'matched_selected', 'intervals', 'paired_contrasts', 'bootstrap_values', 'valid_draw_ids')
    tables = {name: [] for name in names}
    samples = {}
    pair_keys = sorted({u + '__' + v for u, v, _, _ in JOINT})
    pair_indices = {key: [c for c, (u,v,_,_) in enumerate(JOINT) if u+'__'+v == key] for key in pair_keys}
    for view, qualify_bit, outcome_bit in VIEWS:
        qualification = (flags & qualify_bit) != 0
        outcome = (flags & outcome_bit) != 0
        own_e = qualification[:, :, ANCHORS[:, 0]] & qualification[:, :, ANCHORS[:, 1]]
        own_s = qualification[:, :, :41].all(axis=2)
        common_e, common_s = own_e.all(axis=0), own_s.all(axis=0)
        view_candidates = []
        for mi, model in enumerate(models):
            joint = outcome[mi, :, 41:]
            all_survive = joint.all(axis=1)
            for cohort in ('own', 'common4'):
                eligible = own_e[mi] if cohort == 'own' else common_e
                strict = own_s[mi] if cohort == 'own' else common_s
                failed = eligible & ~joint
                ec, fc = eligible.sum(axis=0), failed.sum(axis=0)
                total_e, total_f = int(ec.sum()), int(fc.sum())
                ns = int(strict.sum())
                survive = strict & all_survive
                profile = cell_profile(ec, fc)
                clean = float(outcome[mi, :, 0].mean())
                joint_acc = float(joint.mean())
                base = dict(view=view, cohort=cohort, model=model)
                summary = dict(**base, full_roots=n, full_images=g,
                    clean_correct=int(outcome[mi,:,0].sum()), clean_denominator=n, clean_accuracy=clean,
                    single_correct=int(outcome[mi,:,1:41].sum()), single_denominator=n*40, single_accuracy=float(outcome[mi,:,1:41].mean()),
                    joint_correct=int(joint.sum()), joint_denominator=n*400, joint_accuracy=joint_acc,
                    relative_joint_drop=(clean-joint_acc)/clean if clean else None,
                    joint_clean_retention=joint_acc/clean if clean else None,
                    minimum_joint_cell_accuracy=float(joint.mean(axis=0).min()),
                    eligible_records=total_e, failure_records=total_f,
                    eligible_roots=int(eligible.any(axis=1).sum()), eligible_images=len(set(gids[eligible.any(axis=1)])),
                    ECR=total_e/(n*400), JSR=rate(total_e-total_f,total_e), JFR=rate(total_f,total_e),
                    strict_roots=ns, strict_images=len(set(gids[strict])), strict_root_retention=ns/n,
                    strict_survivors=int(survive.sum()), strict_affected_roots=int((strict & ~all_survive).sum()),
                    JSS=rate(int(survive.sum()),ns), strict_joint_correct=int(joint[strict].sum()), strict_joint_denominator=ns*400,
                    strict_joint_accuracy=rate(int(joint[strict].sum()),ns*400),
                    TGR5=profile.pop('TGR'), **profile)
                tables['summary'].append(summary)
                valid = ec >= 30
                rates = fc[valid]/ec[valid]
                for min_support in (10,30,50):
                    for q in (.01,.05,.10):
                        tables['tail_sensitivity'].append(dict(**base, min_support=min_support, tail_fraction=q,
                                                              **cell_profile(ec,fc,q,min_support)))
                for tau in np.arange(1001)/1000:
                    tables['H_curve'].append(dict(**base, min_support=30, valid_cells=int(valid.sum()), tau=float(tau), H=hotspot(rates,tau)))
                for tau in (.005,.01,.02,.05):
                    tables['H_thresholds'].append(dict(**base, min_support=30, valid_cells=int(valid.sum()), tau=tau,H=hotspot(rates,tau)))
                for c, (u,v,i,j) in enumerate(JOINT):
                    tables['cells'].append(dict(**base, condition=conditions[41+c], image_op=u, text_op=v, image_strength=i,text_strength=j,
                        eligible_roots=int(ec[c]), eligible_images=len(set(gids[eligible[:,c]])), failure_roots=int(fc[c]),
                        failure_images=len(set(gids[failed[:,c]])), conditional_failure=rate(int(fc[c]),int(ec[c])),
                        valid_support30=bool(valid[c]), ECR=float(ec[c]/n), full_joint_correct=int(joint[:,c].sum()), full_roots=n,
                        full_joint_accuracy=float(joint[:,c].mean())))
                source_rows=[]
                for source in source_roster:
                    use = sources == source
                    ss = strict & use
                    se = eligible[use].sum(axis=0)
                    sf = failed[use].sum(axis=0)
                    sr = dict(**base,source=str(source), full_roots=int(use.sum()), full_images=len(set(gids[use])),
                        eligible_records=int(se.sum()), failure_records=int(sf.sum()),
                        eligible_roots=int(eligible[use].any(axis=1).sum()),eligible_images=len(set(gids[use][eligible[use].any(axis=1)])),
                        strict_roots=int(ss.sum()), strict_images=len(set(gids[ss])), strict_survivors=int(survive[use].sum()),
                        JSR=rate(int(se.sum()-sf.sum()),int(se.sum())), JSS=rate(int(survive[use].sum()),int(ss.sum())),
                        clean_accuracy=float(outcome[mi,use,0].mean()), single_accuracy=float(outcome[mi,use,1:41].mean()),joint_accuracy=float(joint[use].mean()))
                    tables['source_summary'].append(sr)
                    source_rows.append(sr)
                    for c in range(400):
                        tables['source_cells'].append(dict(**base,source=str(source), condition=conditions[41+c],
                            source_roots=int(use.sum()), eligible_roots=int(se[c]),eligible_images=len(set(gids[use & eligible[:,c]])),
                            failure_roots=int(sf[c]),conditional_failure=rate(int(sf[c]),int(se[c]))))
                tables['source_equal'].append(dict(**base,source_count=len(source_roster),
                    **{key:_source_equal(source_rows,key) for key in ('JSR','JSS','clean_accuracy','single_accuracy','joint_accuracy')}))
                for ri, root in enumerate(roots):
                    tables['cohort_roots'].append(dict(**base,root=root['m3_id'],image_group=root['image_fingerprint'],source=root['source_dataset'],
                        strict41=bool(strict[ri]),strict_survivor=bool(survive[ri]),eligible_cells=int(eligible[ri].sum()),failure_cells=int(failed[ri].sum())))
                if cohort == 'common4':
                    eg,fg = _group_sum(eligible,gids,g),_group_sum(failed,gids,g)
                    sg,pg = _group_sum(strict,gids,g),_group_sum(survive,gids,g)
                    bs = cluster_bootstrap(eg,fg,sg,pg,draws,valid,weights)
                    for metric, vals in bs.items():
                        samples[(view,model,metric)] = vals
                        tables['intervals'].append(dict(**base,metric=metric,point=summary[metric],**_interval(vals)))
                        for draw_id, val in enumerate(vals):
                            good = bool(np.isfinite(val))
                            tables['bootstrap_values'].append(dict(view=view,model=model,metric=metric,draw_id=draw_id,value=float(val) if good else None,valid=good))
                            if good:
                                tables['valid_draw_ids'].append(dict(view=view,kind='model',key=model,metric=metric,draw_id=draw_id))
            # Each observed example uses common strict41 and exactly 16 cells.
            strict = common_s
            ns = int(strict.sum())
            pair_rows = {}
            for key in pair_keys:
                ix = pair_indices[key]
                correct = int(joint[strict][:,ix].sum())
                survived = int(joint[strict][:,ix].all(axis=1).sum())
                row = dict(view=view,model=model,pair=key,strict_roots=ns,strict_images=len(set(gids[strict])),
                           correct=correct,denominator=ns*16,mean_accuracy=rate(correct,ns*16),survivors=survived,pair_suite_survival=rate(survived,ns))
                tables['pair_suites'].append(row)
                pair_rows[key] = row
                for c in ix:
                    tables['pair_cells'].append(dict(view=view,model=model,pair=key,condition=conditions[41+c],strict_roots=ns,
                        failure_roots=int((~joint[strict,c]).sum()),accuracy=rate(int(joint[strict,c].sum()),ns)))
                for source in source_roster:
                    ss = strict & (sources == source)
                    sn = int(ss.sum())
                    pc = int(joint[ss][:,ix].sum())
                    ps = int(joint[ss][:,ix].all(axis=1).sum())
                    tables['pair_sources'].append(dict(view=view,model=model,pair=key,source=str(source),strict_roots=sn,strict_images=len(set(gids[ss])),
                        correct=pc,denominator=sn*16,mean_accuracy=rate(pc,sn*16),survivors=ps,pair_suite_survival=rate(ps,sn)))
            for a,b in combinations(pair_keys,2):
                ra,rb = pair_rows[a],pair_rows[b]
                delta = abs(ra['correct']-rb['correct'])
                sd = abs(ra['survivors']-rb['survivors'])
                supported = ns>0
                view_candidates.append(dict(view=view,model=model,pair_a=a,pair_b=b,strict_roots=ns,denominator=ns*16,
                    correct_a=ra['correct'],correct_b=rb['correct'],survivors_a=ra['survivors'],survivors_b=rb['survivors'],
                    mean_accuracy_a=ra['mean_accuracy'],mean_accuracy_b=rb['mean_accuracy'],
                    pair_suite_survival_a=ra['pair_suite_survival'],pair_suite_survival_b=rb['pair_suite_survival'],
                    absolute_accuracy_difference=rate(delta,ns*16),absolute_survival_difference=rate(sd,ns),
                    exact_equal_correct=bool(supported and delta==0),supported=supported,
                    included=bool(supported and 100*delta<=ns*16),selected=False,selection_rank=None))
        included = [r for r in view_candidates if r['included']]
        included.sort(key=lambda r:(not r['exact_equal_correct'],-r['absolute_survival_difference'],r['absolute_accuracy_difference'],r['model'],r['pair_a'],r['pair_b']))
        for rank,row in enumerate(included[:3],1):
            row.update(selected=True,selection_rank=rank)
            tables['matched_selected'].append(dict(row))
        tables['matched_candidates'].extend(view_candidates)
        for a,b in combinations(sorted(models),2):
            for metric in METRICS:
                difference = samples[(view,a,metric)]-samples[(view,b,metric)]
                key=a+'__minus__'+b
                points={r['model']:r[metric] for r in tables['summary'] if r['view']==view and r['cohort']=='common4'}
                point = points[a]-points[b] if points[a] is not None and points[b] is not None else None
                tables['paired_contrasts'].append(dict(view=view,model_a=a,model_b=b,metric=metric,point_difference=point,**_interval(difference)))
                for draw_id in np.flatnonzero(np.isfinite(difference)):
                    tables['valid_draw_ids'].append(dict(view=view,kind='contrast',key=key,metric=metric,draw_id=int(draw_id)))
    # Required deterministic identities: fixed Task eligibility, semantic superset.
    for cohort in ('own','common4'):
        for model in models:
            task = next(r for r in tables['summary'] if (r['view'],r['cohort'],r['model'])==('Task',cohort,model))
            fixed = next(r for r in tables['summary'] if (r['view'],r['cohort'],r['model'])==('Semantic-on-fixed-Task',cohort,model))
            assert task['eligible_records']==fixed['eligible_records'] and task['strict_roots']==fixed['strict_roots']
            for metric, sign in (('JSR',1),('JSS',1),('TGR5',-1)):
                if task[metric] is not None:
                    assert sign*(fixed[metric]-task[metric])>=-1e-14
    return dict(tables=tables, draws=draws, metadata=dict(models=list(models),views=[v[0] for v in VIEWS],
        roots=n,image_groups=g,image_roster=image_roster,source_roster=[str(s) for s in source_roster],
        bootstrap=bootstrap,seed=seed,rng='numpy.random.default_rng/PCG64',draw_size=g,
        ci='pointwise descriptive percentile 95%, NumPy linear, valid draws only',
        tail='ceil(q*V), fixed original eligible-root>=30 cell set; any zero cell makes whole draw NA',
        synthetic_examples=synthetic_examples(),fixed_semantic_monotonicity='PASS'))


def _dump(path, value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def run(data_dir, output, bootstrap=2000, seed=17, protocol=None):
    """Authenticate frozen inputs, preserve pre-computation and failure receipts."""
    if bootstrap < 0:
        raise ValueError('bootstrap must be nonnegative')
    protocol = Path(protocol) if protocol else Path(__file__).resolve().parents[3]/'frozen_protocol_v2.md'
    protocol_pin = pin(protocol)
    if protocol_pin['sha256'] != PROTOCOL_SHA256:
        raise ValueError('Frozen protocol SHA256 mismatch')
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    src=Path(data_dir).resolve()
    _dump(output/'run_manifest.json',dict(run_id=output.name,status='started',started_utc=datetime.now(timezone.utc).isoformat(),
        code=pin(__file__),loader=pin(Path(__file__).with_name('reproduce_binary.py')),protocol=protocol_pin,protocol_path=str(protocol.resolve()),
        environment=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,platform=platform.platform(),
                         thread_pins={v:os.environ.get(v) for v in THREAD_VARS}),
        bootstrap=bootstrap,seed=seed,hyperparameters=dict(tail_fractions=[.01,.05,.1],minimum_root_supports=[10,30,50],
                        H_grid_step=.001,H_thresholds=[.005,.01,.02,.05],matched_absolute_accuracy_tolerance=.01),
        data_role='existing public known benchmark; all root/image roles unchanged; no unseen test generalization',
        source_score='Task semantic AND format; completed flags only',model_revision='released four-model labels; no model loaded',
        expected_inputs=PINS,source_dir=str(src),model_calls=0,judge_calls=0,new_input_exposures=0,training=False))
    try:
        _dump(output/'input_manifest.json',{name:pin(src/name) for name in PINS})
        flags,models,conditions,roots=load_release(src)
        _dump(output/'validated_inputs.json',dict(shape=list(flags.shape),models=models,root_ids=[r['m3_id'] for r in roots],
            image_roster=sorted({r['image_fingerprint'] for r in roots}),source_counts={s:sum(r['source_dataset']==s for r in roots) for s in sorted({r['source_dataset'] for r in roots})},
            records_read=int(flags.size),new_model_exposures=0,data_roles_unchanged=True,conditions=conditions))
        result=analyze(flags,models,conditions,roots,bootstrap,seed)
        outputs={}
        for name,rows in result['tables'].items():
            dest=output/(name+'.csv')
            with dest.open('x',encoding='utf-8-sig',newline='') as stream:
                # Empty selected list is valid; schema inherits the candidate table.
                fields=list(rows[0]) if rows else (list(result['tables']['matched_candidates'][0]) if name=='matched_selected' else [])
                writer=csv.DictWriter(stream,fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            outputs[dest.name]=dict(**pin(dest),rows=len(rows),columns=fields)
        np.savez_compressed(output/'bootstrap_draws.npz',draws=result['draws'],image_roster=np.array(result['metadata']['image_roster']))
        outputs['bootstrap_draws.npz']=pin(output/'bootstrap_draws.npz')
        _dump(output/'analysis_metadata.json',result['metadata'])
        outputs['analysis_metadata.json']=pin(output/'analysis_metadata.json')
        final=dict(status='completed',run_id=output.name,seconds=time.monotonic()-start,outputs=outputs,
                   model_calls=0,judge_calls=0,new_input_exposures=0,source_masks_modified=False,
                   stopped_reason='all frozen computations and exports completed')
        _dump(output/'result.json',final)
        return final
    except Exception as exc:
        _dump(output/'failure.json',dict(status='failed',seconds=time.monotonic()-start,error_type=type(exc).__name__,error=str(exc),
                                      stopped_reason='exception; partial outputs preserved'))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--protocol',type=Path)
    args=parser.parse_args()
    result=run(args.data,args.output,args.bootstrap,args.seed,args.protocol)
    print(json.dumps({k:result[k] for k in ('status','run_id','seconds')}))


if __name__=='__main__':
    main()
