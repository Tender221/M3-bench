"""CPU-only replay of all 12 legacy binary tables from the public masks and roots.

No expected CSV, profile, answers, model, network, or GPU dependency is read.
The loader authenticates the frozen release; the pure table API also accepts
small synthetic root/model populations using the same exact 441 coordinates.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

MODELS = ('blip2', 'llava', 'qwen25', 'qwen3vl')
IMAGES = ('gaussian_blur', 'gaussian_noise', 'jpeg', 'brightness', 'contrast')
TEXTS = ('typo', 'keyboard', 'ocr', 'whitespace_punc', 'casing')
AXES = ('semantic_correct', 'format_compliant', 'task_pass')
JOINT = tuple((u, v, i, j) for u in IMAGES for v in TEXTS for i in range(1, 5) for j in range(1, 5))
CONDITIONS = ('clean',) + tuple(f'image__{u}__{i}' for u in IMAGES for i in range(1, 5))
CONDITIONS += tuple(f'text__{v}__{j}' for v in TEXTS for j in range(1, 5))
CONDITIONS += tuple(f'mixed__{u}_{v}__{i}_{j}' for u, v, i, j in JOINT)
ANCHORS = np.array([(1 + IMAGES.index(u)*4 + i-1, 21 + TEXTS.index(v)*4 + j-1)
                    for u, v, i, j in JOINT], dtype=int)
PINS = {
    'binary_masks.npz': '3d7617bdb468bc9baa7e4491f9733f2dfd18b697e41c7828dc480c87cf566c17',
    'roots.csv': '79544771505c3bb2219599d1ad70706794779d81e658b38dbe868dcb11208ff6',
}
TABLE_NAMES = ('model_summary', 'accuracy_axes', 'cell_metrics', 'pair_metrics', 'root_metrics',
               'source_metrics', 'slice_coverage', 'missed_locations', 'recurrence', 'budget_curves',
               'model_replacement', 'source_equal_coverage')
ROOT_FIELDS = ('m3_id', 'image_fingerprint', 'source_dataset', 'source_id', 'source_image_id', 'question_type')


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'sha256': digest, 'bytes': path.stat().st_size}


def rate(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _names(values, label):
    values = tuple(values)
    if not values or any(not isinstance(v, str) or not v.strip() or v != v.strip() or '::' in v for v in values):
        raise ValueError(f'Invalid {label}')
    if len(set(values)) != len(values):
        raise ValueError(f'Duplicate {label}')
    return values


def validate_masks(flags, models, conditions, roots):
    """Reject missing/compressed/error labels and noncanonical coordinate mappings."""
    models = _names(models, 'models')
    conditions = _names(conditions, 'conditions')
    if conditions != CONDITIONS:
        raise ValueError('Expected exact ordered clean+40-single+400-joint coordinate IDs')
    if not roots or any(any(k not in r or not isinstance(r[k], str) for k in ROOT_FIELDS) for r in roots):
        raise ValueError('Missing canonical root metadata')
    _names([r['m3_id'] for r in roots], 'root IDs')
    if any(not r['image_fingerprint'] or not r['source_dataset'] for r in roots):
        raise ValueError('Empty image/source identity')
    if not isinstance(flags, np.ndarray) or flags.dtype.kind not in 'ui' or flags.dtype.kind == 'b':
        raise ValueError('Flags must be an integer NumPy array')
    if flags.shape != (len(models), len(roots), 441):
        raise ValueError('Mask shape differs from IDs')
    # bit8=completed; bit1=semantic; bit2=format; bit4=task=semantic AND format.
    if not np.isin(flags, (8, 9, 10, 15)).all():
        raise ValueError('Invalid completion bit, unknown bits, or task conjunction')


def load_release(data_dir):
    """Return (flags, model IDs, condition IDs, canonical root dicts), physically pinned.

    Root metadata aliases: root -> m3_id, image_group -> image_fingerprint,
    source -> source_dataset. IDs/type remain strings, including empty source_image_id.
    """
    src = Path(data_dir)
    for name, expected in PINS.items():
        if pin(src / name)['sha256'] != expected:
            raise ValueError(f'Frozen input SHA256 mismatch: {name}')
    with (src / 'roots.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['root', 'image_group', 'source', 'source_id', 'source_image_id', 'question_type']:
            raise ValueError('Unexpected roots.csv schema')
        roots = [dict(zip(ROOT_FIELDS, [r[k] for k in reader.fieldnames])) for r in reader]
    with np.load(src / 'binary_masks.npz', allow_pickle=False) as data:
        if set(data.files) != {'flags', 'models', 'conditions', 'root_ids'}:
            raise ValueError('Unexpected NPZ members')
        flags = data['flags'].copy()
        models = data['models'].tolist()
        conditions = data['conditions'].tolist()
        root_ids = data['root_ids'].tolist()
    validate_masks(flags, models, conditions, roots)
    if tuple(models) != MODELS or len(roots) != 1500 or len({r['image_fingerprint'] for r in roots}) != 1065:
        raise ValueError('Frozen population differs from 4 models/1500 roots/1065 images')
    if root_ids != [r['m3_id'] for r in roots]:
        raise ValueError('NPZ root IDs do not match roots.csv order')
    return flags, models, conditions, roots


def _gap_stats(eligible, failed, images):
    """eligible/failed are root x cell arrays; images follows selected root order."""
    e, f = int(eligible.sum()), int(failed.sum())
    er, fr = eligible.any(axis=1), failed.any(axis=1)
    total = int(eligible.size)
    # Preserve the alphabetic profile-field schema used by the historical CSVs.
    return dict(eligible=e, eligible_images=len(set(images[er])), eligible_rate=rate(e, total),
                eligible_roots=int(er.sum()), gap=f, gap_images=len(set(images[fr])), gap_rate=rate(f, e),
                gap_roots=int(fr.sum()), gap_total_rate=rate(f, total), total=total,
                total_images=len(set(images)), total_roots=len(images))


def expected_uniform(k, budget, total=400):
    """Hypergeometric probability of finding >=1 failure without replacement."""
    if budget > total - k:
        return 1.0
    return -math.expm1(math.lgamma(total-k+1)-math.lgamma(total-k-budget+1)
                       -math.lgamma(total+1)+math.lgamma(total-budget+1))


def tables_from_masks(flags, models, conditions, roots):
    """Reconstruct 12 legacy table-name -> list[dict] entries from paired masks.

    Fractions are in [0,1]; undefined rates are None (empty CSV field). The legacy
    source_equal_coverage means average within-source *affected-question rate*.
    All qualifications are Task-based. Exact-cell eligibility does not require
    clean correctness; strict41 requires clean and every single condition.
    """
    validate_masks(flags, models, conditions, roots)
    tables = {name: [] for name in TABLE_NAMES}
    n = len(roots)
    images = np.array([r['image_fingerprint'] for r in roots])
    sources = np.array([r['source_dataset'] for r in roots])
    groups = sorted(set(sources))
    masks, failures, suite = {}, {}, {}
    diagonal = np.array([i == j for u, v, i, j in JOINT])
    corner = np.array([i == j == 4 for u, v, i, j in JOINT])
    maxorder = sorted(range(400), key=lambda z: (-JOINT[z][2]-JOINT[z][3], -min(JOINT[z][2:]), -JOINT[z][2], conditions[41+z]))
    diagorder = sorted(range(400), key=lambda z: (not diagonal[z], -JOINT[z][2]-JOINT[z][3], -min(JOINT[z][2:]), -JOINT[z][2], conditions[41+z]))
    for mi, model in enumerate(models):
        grid = (flags[mi] & 4) > 0
        strict = grid[:, :41].all(axis=1)
        fail = ~grid[:, 41:]
        masks[model], failures[model], suite[model] = strict, fail, grid.all(axis=1)
        eligible = grid[:, ANCHORS[:, 0]] & grid[:, ANCHORS[:, 1]]
        exact_gap = eligible & fail
        for axis, bit in zip(AXES, (1, 2, 4)):
            g = (flags[mi] & bit) > 0
            for category, indices in dict(clean=[0], image=range(1, 21), text=range(21, 41), single=range(1, 41), joint=range(41, 441), all=range(441)).items():
                passed = int(g[:, indices].sum())
                total = n * len(indices)
                tables['accuracy_axes'].append(dict(model=model, axis=axis, subset=category, correct=passed, denominator=total, accuracy=rate(passed, total)))
        q = int(strict.sum())
        failed = fail[strict]
        k = failed.sum(axis=1)
        affected = k > 0
        clean, joint_acc = float(grid[:, 0].mean()), float(grid[:, 41:].mean())
        tables['model_summary'].append(dict(model=model, roots=n, images=len(set(images)), clean_correct=int(grid[:, 0].sum()), clean_accuracy=clean,
            single_correct=int(grid[:, 1:41].sum()), single_denominator=n*40, single_accuracy=float(grid[:, 1:41].mean()),
            joint_correct=int(grid[:, 41:].sum()), joint_denominator=n*400, joint_accuracy=joint_acc,
            binary_retention=rate(joint_acc, clean), strict41_roots=q, strict41_images=len(set(images[strict])),
            affected_strict41_roots=int(affected.sum()), affected_fraction=rate(int(affected.sum()), q),
            affected_strict41_images=len(set(images[strict][affected])), strict41_failure_events=int(k.sum()),
            strict41_event_density=rate(int(k.sum()), q*400), median_events_affected=float(np.median(k[affected])) if affected.any() else None,
            single_cell_only_roots=int((k == 1).sum()), all_suite_pass_roots=int(suite[model].sum()),
            all_exact_gap_events=int(exact_gap.sum()), all_exact_gap_eligible=int(eligible.sum()), all_exact_gap_rate=rate(int(exact_gap.sum()), int(eligible.sum()))))
        for threshold in (1, 2, 5, 10, 20):
            tables['recurrence'].append(dict(model=model, threshold=threshold, qualified_roots=q, affected_roots=int((k >= threshold).sum())))
        source_subset = []
        for source in groups:
            use = sources == source
            sq = use & strict
            row = dict(model=model, source=str(source), roots=int(use.sum()), clean_accuracy=float(grid[use, 0].mean()), joint_accuracy=float(grid[use, 41:].mean()),
                       qualified_roots=int(sq.sum()), affected_roots=int((fail.any(axis=1) & sq).sum()),
                       suite_failure_rate=rate(int((fail.any(axis=1) & sq).sum()), int(sq.sum())))
            tables['source_metrics'].append(row)
            source_subset.append(row)
        valid = [r['suite_failure_rate'] for r in source_subset if r['suite_failure_rate'] is not None]
        tables['source_equal_coverage'].append(dict(model=model, source_count=len(groups), valid_sources=len(valid),
            source_equal_coverage=sum(valid)/len(groups) if len(valid) == len(groups) else None,
            pooled_qualified=q, pooled_affected=int(affected.sum()), interval='not_computed'))
        for ri, root in enumerate(roots):
            tables['root_metrics'].append(dict(model=model, root=root['m3_id'], image_group=root['image_fingerprint'], source=root['source_dataset'], clean_pass=bool(grid[ri, 0]),
                strict41=bool(strict[ri]), full_suite_pass=bool(suite[model][ri]), joint_failures=int(fail[ri].sum()),
                diagonal_failure=bool(fail[ri, diagonal].any()), corner_failure=bool(fail[ri, corner].any())))
        for stratum, use in (('all', np.ones(n, dtype=bool)), ('clean_pass', grid[:, 0]), ('strict41', strict)):
            sg = grid[use]
            sf = flags[mi, use]
            se, sx = eligible[use], exact_gap[use]
            si = images[use]
            for u, v in sorted(((u, v) for u in IMAGES for v in TEXTS), key=lambda uv: '_'.join(uv)):
                ix = [z for z, cell in enumerate(JOINT) if cell[:2] == (u, v)]
                tables['pair_metrics'].append(dict(model=model, stratum=stratum, image_op=u, text_op=v, **_gap_stats(se[:, ix], sx[:, ix], si)))
            denominator = len(sg)
            clean_p = rate(int(sg[:, 0].sum()), denominator)
            for z, (u, v, i, j) in enumerate(JOINT):
                ii, ti = ANCHORS[z]
                image_p, text_p = rate(int(sg[:, ii].sum()), denominator), rate(int(sg[:, ti].sum()), denominator)
                joint_p = rate(int(sg[:, 41+z].sum()), denominator)
                excess = (image_p*text_p/clean_p-joint_p)/clean_p if clean_p else None
                stats = _gap_stats(se[:, z:z+1], sx[:, z:z+1], si)
                tables['cell_metrics'].append(dict(model=model, stratum=stratum, image_op=u, text_op=v, image_strength=i, text_strength=j, **stats,
                    task_accuracy=joint_p, semantic_accuracy=rate(int(((sf[:, 41+z] & 1) > 0).sum()), denominator),
                    format_rate=rate(int(((sf[:, 41+z] & 2) > 0).sum()), denominator), clean_task_accuracy=clean_p,
                    image_task_accuracy=image_p, text_task_accuracy=text_p, normalized_product_excess=excess))
        for name, selected in (('full', np.ones(400, dtype=bool)), ('diagonal', diagonal), ('maximum_corner', corner)):
            found = failed[:, selected].any(axis=1)
            missed = affected & ~found
            tables['slice_coverage'].append(dict(model=model, slice=name, conditions=int(selected.sum()), qualified_roots=q,
                full_affected_roots=int(affected.sum()), found_roots=int(found.sum()), missed_roots=int(missed.sum()), discovery_fraction=rate(int(found.sum()), int(affected.sum()))))
            if name != 'full':
                for i in range(1, 5):
                    for j in range(1, 5):
                        selected_ij = np.array([a == i and b == j for u, v, a, b in JOINT])
                        tables['missed_locations'].append(dict(model=model, slice=name, image_strength=i, text_strength=j, missed_roots=int(missed.sum()),
                            failing_roots_at_coordinate=int(failed[missed][:, selected_ij].any(axis=1).sum())))
        for budget in range(1, 401):
            for policy, found in (('uniform_expected', sum(expected_uniform(int(x), budget) for x in k)),
                                  ('maximum_first', int(failed[:, maxorder[:budget]].any(axis=1).sum())),
                                  ('diagonal_first', int(failed[:, diagorder[:budget]].any(axis=1).sum()))):
                tables['budget_curves'].append(dict(model=model, policy=policy, budget=budget, qualified_roots=q, full_affected_roots=int(affected.sum()),
                    discovered_roots=found, discovery_fraction=rate(found, int(affected.sum()))))
    replacement = tables['model_replacement']
    for old in models:
        for ni, new in enumerate(models):
            if old == new:
                continue
            g = (flags[ni] & 4) > 0
            for name, cohort in (('old_full_suite_pass', suite[old]), ('old_strict41', masks[old])):
                count = int(cohort.sum())
                parts = (cohort & ~g[:, 0], cohort & g[:, 0] & ~g[:, 1:41].all(axis=1),
                         cohort & g[:, :41].all(axis=1) & ~g[:, 41:].all(axis=1), cohort & g.all(axis=1))
                counts = [int(part.sum()) for part in parts]
                if sum(counts) != count:
                    raise ValueError('Fixed-population replacement partition is not exhaustive')
                replacement.append(dict(old_model=old, new_model=new, cohort=name, denominator=count, clean_regression=counts[0], single_regression=counts[1],
                    joint_regression=counts[2], preserved=counts[3], preservation_rate=rate(counts[3], count)))
            common = masks[old] & masks[new]
            of, nf = failures[old][common], failures[new][common]
            replacement.append(dict(old_model=old, new_model=new, cohort='common_strict41', denominator=int(common.sum()), clean_regression=0, single_regression=0,
                joint_regression=int((~of.any(axis=1) & nf.any(axis=1)).sum()), preserved=int((~of.any(axis=1) & ~nf.any(axis=1)).sum()), preservation_rate=None,
                repaired_roots=int((of.any(axis=1) & ~nf.any(axis=1)).sum()), persistent_roots=int((of.any(axis=1) & nf.any(axis=1)).sum()),
                repaired_conditions=int((of & ~nf).sum()), new_failed_conditions=int((~of & nf).sum())))
    fields = ('old_model', 'new_model', 'cohort', 'denominator', 'clean_regression', 'single_regression', 'joint_regression', 'preserved', 'preservation_rate',
              'repaired_roots', 'persistent_roots', 'repaired_conditions', 'new_failed_conditions')
    tables['model_replacement'] = [{k: row.get(k) for k in fields} for row in replacement]
    return tables


def _dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def reproduce(data_dir, output, protocol=None):
    """Write only into a fresh directory; preserve pre-computation/failure receipts."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    src = Path(data_dir).resolve()
    manifest = dict(run_id=output.name, status='started', started_utc=datetime.now(timezone.utc).isoformat(),
        code=pin(__file__), code_path=str(Path(__file__).resolve()), protocol='legacy-binary-12-table-mask-replay-v1',
        protocol_pin=pin(protocol) if protocol else None, source_dir=str(src),
        expected_input_sha256=PINS, environment=dict(executable=sys.executable, python=sys.version, numpy=np.__version__, platform=platform.platform()),
        seed=None, randomness='none', data_role='existing public known benchmark; fixed root/image identities',
        source_score='task_pass = semantic_correct AND format_compliant; completion bit required',
        input_exposure='existing masks only, no new input exposure', model_revision='existing released model masks; no model loaded',
        training=False, model_calls=0, judge_calls=0, expected_tables=list(TABLE_NAMES))
    # Receipt exists even if inputs are absent or fail authentication.
    _dump(output / 'run_manifest.json', manifest)
    try:
        actual = {name: pin(src / name) for name in PINS}
        _dump(output / 'input_manifest.json', dict(physical_inputs=actual, expected_sha256=PINS))
        flags, models, conditions, roots = load_release(src)
        _dump(output / 'validated_inputs.json', dict(shape=list(flags.shape), models=models, conditions=conditions,
            roots=len(roots), image_groups=len({r['image_fingerprint'] for r in roots}), scored_entries=int(flags.size),
            root_ids=[r['m3_id'] for r in roots], data_roles='unchanged existing benchmark',
            source_counts={s: sum(r['source_dataset'] == s for r in roots) for s in sorted({r['source_dataset'] for r in roots})},
            anchor_mapping=[dict(condition=conditions[41+z], image_anchor=conditions[ii], text_anchor=conditions[ti]) for z, (ii, ti) in enumerate(ANCHORS)]))
        tables = tables_from_masks(flags, models, conditions, roots)
        outputs = {}
        for name, rows in tables.items():
            if not rows:
                raise ValueError(f'Empty released table: {name}')
            dest = output / f'{name}.csv'
            with dest.open('x', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            outputs[dest.name] = dict(**pin(dest), rows=len(rows), columns=list(rows[0]))
        result = dict(status='completed', run_id=output.name, seconds=time.monotonic()-started, tables=outputs,
                      source_masks_modified=False, expected_csvs_read=False, model_calls=0, judge_calls=0)
        _dump(output / 'result.json', result)
        return result
    except Exception as exc:
        _dump(output / 'failure.json', dict(status='failed', seconds=time.monotonic()-started, error_type=type(exc).__name__, error=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', '--input', dest='data_dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    result = reproduce(args.data_dir, args.output, args.protocol)
    print(json.dumps({k: result[k] for k in ('status', 'run_id', 'seconds')}))


if __name__ == '__main__':
    main()
