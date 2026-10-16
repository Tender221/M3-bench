"""Fast CLI boundary tests; the full frozen-data acceptance is separate."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

CODE = Path(__file__).resolve().parents[1]
SCRIPT = CODE / 'reproduce_all.py'


class UnifiedEntryTests(unittest.TestCase):
    def call(self, *args, cwd=None):
        self.assertTrue(SCRIPT.is_file(), 'Unified runnable entry is not implemented')
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=cwd,
                              capture_output=True, text=True)

    def test_help_from_clean_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.call('--help', cwd=tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            for flag in ('--data', '--output', '--plots', '--bootstrap', '--seed', '--protocol'):
                self.assertIn(flag, result.stdout)

    def test_missing_protocol_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'fresh'
            result = self.call('--output', out, '--protocol', Path(tmp)/'absent.md')
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out.exists())

    def test_changed_protocol_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol = Path(tmp)/'changed.md'
            protocol.write_text('changed frozen definitions', encoding='utf-8')
            out = Path(tmp)/'fresh'
            result = self.call('--output', out, '--protocol', protocol)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('protocol', result.stderr.lower())
            self.assertFalse(out.exists())

    def test_existing_output_preserves_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'existing'
            out.mkdir()
            marker = out/'keep.txt'
            marker.write_text('untouched')
            result = self.call('--output', out)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(marker.read_text(), 'untouched')
            self.assertEqual([p.name for p in out.iterdir()], ['keep.txt'])

    def test_tampered_inputs_preserve_failure_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)/'data'
            data.mkdir()
            # Reject bytes by the frozen physical pin, before NPZ parsing.
            (data/'binary_masks.npz').write_bytes(b'tampered')
            (data/'roots.csv').write_text('tampered')
            out = Path(tmp)/'fresh'
            result = self.call('--data', data, '--output', out, cwd=tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((out/'run_manifest.json').is_file())
            self.assertTrue((out/'failure.json').is_file())
            self.assertFalse((out/'joint_metrics_v1').exists())
            self.assertFalse((out/'result.json').exists())

    def test_plotter_is_separate_optional_module(self):
        self.assertTrue((CODE/'plot_joint_metrics.py').is_file(), 'CSV-only plotter is not implemented')


if __name__ == '__main__':
    unittest.main()
