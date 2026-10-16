"""Behavioral tests; synthetic masks use the complete public coordinate ABI."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np

BASE = Path(__file__).resolve().parents[1]
CODE = BASE / 'reproduce_binary.py'
if CODE.exists():
    spec = importlib.util.spec_from_file_location('reproduce_binary', CODE)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
else:
    # Deliberate missing-behavior fallback yields assertions rather than import errors.
    api = types.SimpleNamespace(tables_from_masks=lambda *args: {})

IMAGES = ('gaussian_blur', 'gaussian_noise', 'jpeg', 'brightness', 'contrast')
TEXTS = ('typo', 'keyboard', 'ocr', 'whitespace_punc', 'casing')
CONDS = ['clean'] + [f'image__{u}__{i}' for u in IMAGES for i in range(1, 5)]
CONDS += [f'text__{v}__{j}' for v in TEXTS for j in range(1, 5)]
CONDS += [f'mixed__{u}_{v}__{i}_{j}' for u in IMAGES for v in TEXTS for i in range(1, 5) for j in range(1, 5)]


def fixture(n=4, models=('old', 'new')):
    roots = [dict(m3_id=f'r{k}', image_fingerprint=f'im{k}', source_dataset='a' if k == 0 else 'b',
                  source_id=str(k), source_image_id='', question_type='what') for k in range(n)]
    return np.full((len(models), n, 441), 15, dtype=np.uint8), list(models), CONDS.copy(), roots


def one(rows, **keys):
    matches = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    assert len(matches) == 1, (keys, matches)
    return matches[0]


class ReplayTests(unittest.TestCase):
    def test_all_400_image_and_text_anchors_independently(self):
        flags, models, conds, roots = fixture(n=40, models=('all_anchors',))
        # Root s fails exactly single s+1. Each joint's two failing outcomes
        # are precisely its disqualified roots, so every exact gap count is zero.
        for s in range(40):
            flags[0, s, s+1] = 10
        for z, (u, v, i, j) in enumerate((u, v, i, j) for u in IMAGES for v in TEXTS for i in range(1, 5) for j in range(1, 5)):
            image_root = IMAGES.index(u)*4 + i-1
            text_root = 20 + TEXTS.index(v)*4 + j-1
            flags[0, [image_root, text_root], 41+z] = 10
        tables = api.tables_from_masks(flags, models, conds, roots)
        cells = [r for r in tables['cell_metrics'] if r['stratum'] == 'all']
        self.assertEqual(len(cells), 400)
        self.assertTrue(all((r['eligible'], r['gap']) == (38, 0) for r in cells))

    def test_twelve_tables_and_corresponding_anchors_without_clean(self):
        flags, models, conds, roots = fixture()
        flags[0, 0, 0] = 10  # clean fails; both corresponding singles still pass
        flags[0, 0, 41] = 10
        flags[0, 1, 1] = 10  # only the exact image anchor disqualifies its cells
        tables = api.tables_from_masks(flags, models, conds, roots)
        self.assertEqual(set(tables), {'model_summary', 'accuracy_axes', 'cell_metrics', 'pair_metrics',
            'root_metrics', 'source_metrics', 'slice_coverage', 'missed_locations', 'recurrence',
            'budget_curves', 'model_replacement', 'source_equal_coverage'})
        cell = one(tables['cell_metrics'], model='old', stratum='all', image_op='gaussian_blur',
                   text_op='typo', image_strength=1, text_strength=1)
        self.assertEqual((cell['eligible'], cell['gap']), (3, 1))
        other = one(tables['cell_metrics'], model='old', stratum='all', image_op='gaussian_blur',
                    text_op='typo', image_strength=2, text_strength=1)
        self.assertEqual(other['eligible'], 4)
        strict = one(tables['model_summary'], model='old')
        self.assertEqual(strict['strict41_roots'], 2)
        clean = one(tables['cell_metrics'], model='old', stratum='clean_pass', image_op='gaussian_blur',
                    text_op='typo', image_strength=1, text_strength=1)
        self.assertEqual((clean['total'], clean['eligible'], clean['gap']), (3, 2, 0))

    def test_diagonal_corner_misses_and_image_deduplication(self):
        flags, models, conds, roots = fixture()
        roots[1]['image_fingerprint'] = roots[0]['image_fingerprint']
        flags[0, 0, conds.index('mixed__contrast_casing__1_2')] = 10
        flags[0, 1, conds.index('mixed__jpeg_ocr__3_3')] = 10
        tables = api.tables_from_masks(flags, models, conds, roots)
        diag = one(tables['slice_coverage'], model='old', slice='diagonal')
        corner = one(tables['slice_coverage'], model='old', slice='maximum_corner')
        self.assertEqual((diag['found_roots'], diag['missed_roots']), (1, 1))
        self.assertEqual((corner['found_roots'], corner['missed_roots']), (0, 2))
        self.assertEqual(one(tables['model_summary'], model='old')['affected_strict41_images'], 1)
        missed = one(tables['missed_locations'], model='old', slice='diagonal', image_strength=1, text_strength=2)
        self.assertEqual(missed['failing_roots_at_coordinate'], 1)

    def test_model_replacement_partitions_fixed_old_population(self):
        flags, models, conds, roots = fixture()
        flags[1, 0, 0] = 10
        flags[1, 1, 1] = 10
        flags[1, 2, 41] = 10
        tables = api.tables_from_masks(flags, models, conds, roots)
        row = one(tables['model_replacement'], old_model='old', new_model='new', cohort='old_full_suite_pass')
        self.assertEqual([row[k] for k in ('denominator', 'clean_regression', 'single_regression', 'joint_regression', 'preserved')], [4, 1, 1, 1, 1])
        common = one(tables['model_replacement'], old_model='old', new_model='new', cohort='common_strict41')
        self.assertEqual((common['denominator'], common['joint_regression'], common['new_failed_conditions']), (2, 1, 1))
        self.assertIsNone(common['preservation_rate'])

    def test_source_equal_is_mean_of_source_ratios_not_pooled(self):
        flags, models, conds, roots = fixture()
        flags[0, 0, 41] = 10
        tables = api.tables_from_masks(flags, models, conds, roots)
        row = one(tables['source_equal_coverage'], model='old')
        self.assertEqual(row['source_equal_coverage'], .5)
        self.assertEqual((row['pooled_affected'], row['pooled_qualified']), (1, 4))
        flags[0, 0, 0] = 10
        row = one(api.tables_from_masks(flags, models, conds, roots)['source_equal_coverage'], model='old')
        self.assertIsNone(row['source_equal_coverage'])
        self.assertEqual(row['valid_sources'], 1)

    def test_duplicates_bad_shape_completion_conjunction_and_coordinates(self):
        for defect in ('root', 'model', 'condition', 'shape', 'completion', 'conjunction', 'float', 'coordinate'):
            with self.subTest(defect=defect):
                flags, models, conds, roots = fixture()
                if defect == 'root': roots[1]['m3_id'] = roots[0]['m3_id']
                if defect == 'model': models[1] = models[0]
                if defect == 'condition': conds[1] = conds[2]
                if defect == 'shape': flags = flags[:, :, :-1]
                if defect == 'completion': flags[0, 0, 0] = 7
                if defect == 'conjunction': flags[0, 0, 0] = 11
                if defect == 'float': flags = flags.astype(float)
                if defect == 'coordinate': conds[41], conds[42] = conds[42], conds[41]
                with self.assertRaises(ValueError): api.tables_from_masks(flags, models, conds, roots)

    def test_empty_strict_cohort_and_zero_clean_use_none(self):
        flags, models, conds, roots = fixture()
        flags[0, :, 0] = 10
        tables = api.tables_from_masks(flags, models, conds, roots)
        row = one(tables['model_summary'], model='old')
        self.assertIsNone(row['binary_retention'])
        self.assertIsNone(row['affected_fraction'])
        self.assertIsNone(one(tables['cell_metrics'], model='old', stratum='strict41', image_op='jpeg',
                              text_op='ocr', image_strength=1, text_strength=1)['task_accuracy'])

    def test_fresh_output_refusal_and_physical_pin_rejection(self):
        self.assertTrue(callable(getattr(api, 'reproduce', None)), 'fresh replay writer is missing')
        self.assertTrue(callable(getattr(api, 'load_release', None)), 'pinned public loader is missing')
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            existing = base / 'existing'
            existing.mkdir()
            (existing / 'keep.txt').write_text('unchanged')
            with self.assertRaises(FileExistsError): api.reproduce(base / 'absent', existing)
            self.assertEqual((existing / 'keep.txt').read_text(), 'unchanged')
            (base / 'binary_masks.npz').write_bytes(b'modified')
            (base / 'roots.csv').write_bytes(b'modified')
            with self.assertRaises(ValueError): api.load_release(base)


if __name__ == '__main__':
    unittest.main(verbosity=2)
