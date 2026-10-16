"""Synthetic layout regression: figure headings must not overlap panel titles."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class PlotLayoutTests(unittest.TestCase):
    def test_heading_panel_title_clearance(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest('Optional plotting dependency is absent')
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        code=Path(__file__).resolve().parents[1]/'plot_joint_metrics.py'
        spec=importlib.util.spec_from_file_location('plotter',code)
        api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
        row=dict(view='Task',model='blip2',pair_a='brightness__keyboard',pair_b='contrast__ocr',
                 selected=True,included=True,selection_rank=1,strict_roots=88,denominator=1408,
                 correct_a=1399,correct_b=1399,survivors_a=82,survivors_b=85,
                 mean_accuracy_a=1399/1408,mean_accuracy_b=1399/1408,
                 pair_suite_survival_a=82/88,pair_suite_survival_b=85/88)
        # Explicitly synthetic layout fixture: no claim of empirical selection.
        with tempfile.TemporaryDirectory() as temp:
            data=Path(temp)/'data';data.mkdir()
            with (data/'matched_selected.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            with patch.object(plt,'close'):
                api.plot(data,Path(temp)/'figure')
                fig=plt.gcf();fig.canvas.draw();renderer=fig.canvas.get_renderer()
                for text in fig.texts[:2]:
                    for ax in fig.axes:
                        panel_title=ax._left_title
                        self.assertFalse(text.get_window_extent(renderer).overlaps(panel_title.get_window_extent(renderer)),
                                         'Figure heading overlaps panel title')
            plt.close(fig)


if __name__=='__main__':
    unittest.main()
