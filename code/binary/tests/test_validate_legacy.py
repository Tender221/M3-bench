"""Legacy reference validation must reject category, integer and float drift."""
import importlib.util
import csv
from pathlib import Path
import tempfile
import unittest

HELPER=Path(__file__).resolve().parents[1]/'validate_legacy.py'


class ReferenceValidationTests(unittest.TestCase):
    def test_complete_tables_and_numeric_tampering(self):
        self.assertTrue(HELPER.is_file(),'Post-generation legacy validation is missing')
        spec=importlib.util.spec_from_file_location('validate_legacy',HELPER)
        api=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        with tempfile.TemporaryDirectory() as temp:
            a,e=Path(temp)/'actual',Path(temp)/'expected'
            a.mkdir();e.mkdir()
            for name, keys in api.KEYS.items():
                for directory in (a,e):
                    with (directory/(name+'.csv')).open('w',newline='') as stream:
                        writer=csv.writer(stream)
                        writer.writerow([*keys,'eligible','accuracy'])
                        writer.writerow([*[('x' if k in api.CATEGORICAL else 1) for k in keys],3,.5])
            self.assertEqual(api.compare(a,e)['table_count'],12)
            target=a/'model_summary.csv'
            original=target.read_text()
            for text in (original.replace('x,3,0.5','y,3,0.5'),
                         original.replace('x,3,0.5','x,4,0.5'),
                         original.replace('x,3,0.5','x,3,0.6')):
                target.write_text(text)
                with self.assertRaises(AssertionError):
                    api.compare(a,e)
            target.write_text(original)


if __name__=='__main__':
    unittest.main()
