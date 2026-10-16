import importlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import joint_metrics as jm
from reproduce_binary import CONDITIONS, ANCHORS


def fixture(n=4):
    roots = [dict(m3_id=str(i), image_fingerprint=str(i//2), source_dataset=('a','b','c')[i%3], source_id=str(i), source_image_id='', question_type='q') for i in range(n)]
    return np.full((4,n,441),15,dtype=np.uint8), ['a','b','c','d'], list(CONDITIONS), roots


class FrozenContracts(unittest.TestCase):
    def function(self, name):
        self.assertTrue(callable(getattr(jm,name,None)), f'missing required behavior: {name}')
        return getattr(jm,name)

    def test_equal_marginal_jsr(self):
        examples = self.function('synthetic_examples')()
        a,b=examples['jsr']
        self.assertEqual(a['marginals'],b['marginals'])
        self.assertEqual((a['JSR'],b['JSR']),(1.,0.))

    def test_equal_cell_map_jss(self):
        a,b=self.function('synthetic_examples')()['jss']
        self.assertEqual(a['cell_accuracies'],b['cell_accuracies'])
        self.assertEqual((a['JSS'],b['JSS']),(.5,0.))
        self.assertEqual(len(a['cell_accuracies']),400)

    def test_tail_ceil_support_micro_and_strict_threshold(self):
        f=self.function('cell_profile')
        r=f(np.array([30,30,30,29]),np.array([3,6,9,29]),q=.34,min_support=30)
        self.assertEqual(r['tail_cells'],2)
        self.assertAlmostEqual(r['TGR'],.25)
        self.assertAlmostEqual(r['macro_failure'],.2)
        self.assertAlmostEqual(r['micro_failure_all'],47/119)
        self.assertAlmostEqual(r['micro_failure_valid'],.2)
        self.assertAlmostEqual(f(np.array([30,300]),np.array([3,0]))['micro_failure_valid'],3/330)
        self.assertEqual(self.function('hotspot')(np.array([.1,.2,.3]),.2),1/3)
        self.assertIsNone(f(np.zeros(4,int),np.zeros(4,int))['TGR'])
        self.assertIsNone(f(np.zeros(4,int),np.zeros(4,int))['micro_failure_all'])

    def test_bootstrap_cluster_fixed_validity_and_determinism(self):
        f=self.function('cluster_bootstrap')
        # image 0 contains 30 eligible roots; image 1 contains 1. The original
        # valid cell stays valid at resampled support 2 (<30), but empty -> NA.
        eg=np.array([[30,30],[1,0]],float)
        fg=np.array([[15,0],[1,0]],float)
        draws=np.array([[0,0],[1,1],[0,1]],int)
        result=f(eg,fg,np.array([1,0.]),np.array([1,0.]),draws,np.array([True,True]))
        np.testing.assert_allclose(result['JSR'],[.75,0,1-16/61])
        np.testing.assert_allclose(result['JSS'],[1,np.nan,1],equal_nan=True)
        np.testing.assert_allclose(result['TGR5'],[.5,np.nan,16/31],equal_nan=True)
        again=f(eg,fg,np.array([1,0.]),np.array([1,0.]),draws,np.array([True,True]))
        np.testing.assert_equal(result['TGR5'],again['TGR5'])
        d1=self.function('draw_images')(2,17,20)
        np.testing.assert_equal(d1,jm.draw_images(2,17,20))
        self.assertEqual(d1.shape,(20,2))

    def test_analyze_common_and_fixed_semantic(self):
        f=self.function('analyze')
        flags,models,conditions,roots=fixture()
        flags[0,0,41]=9  # semantic passes, task fails
        flags[1,1,ANCHORS[0,0]]=8
        result=f(flags,models,conditions,roots,bootstrap=12,seed=17)
        rows=result['tables']['summary']
        task=next(x for x in rows if x['view']=='Task' and x['cohort']=='common4' and x['model']=='a')
        fixed=next(x for x in rows if x['view']=='Semantic-on-fixed-Task' and x['cohort']=='common4' and x['model']=='a')
        self.assertEqual(task['strict_roots'],3)
        self.assertEqual(task['strict_survivors'],2)
        self.assertEqual(task['eligible_records'],fixed['eligible_records'])
        self.assertGreaterEqual(fixed['JSR'],task['JSR'])
        self.assertEqual(len(result['tables']['matched_candidates']),3600)
        self.assertEqual(len(result['tables']['cells']),3*2*4*400)
        self.assertEqual(len(result['tables']['intervals']),3*4*3)
        self.assertEqual(len(result['tables']['paired_contrasts']),3*6*3)
        self.assertEqual(len(result['tables']['H_curve']),3*2*4*1001)
        self.assertEqual(len(result['tables']['tail_sensitivity']),3*2*4*9)
        self.assertEqual(len(result['tables']['matched_selected']),9)

    def test_empty_support_and_unsupported_candidates(self):
        flags,models,conditions,roots=fixture()
        flags[:,:,:]=8
        result=self.function('analyze')(flags,models,conditions,roots,bootstrap=3)
        for r in result['tables']['summary']:
            for key in ('JSR','JSS','TGR5','relative_joint_drop','joint_clean_retention'):
                self.assertIsNone(r[key])
        self.assertTrue(all(not r['included'] for r in result['tables']['matched_candidates']))
        self.assertEqual(result['tables']['matched_selected'],[])
        self.assertTrue(all(r['valid_draws']==0 for r in result['tables']['intervals']))

    def test_independent_naive_bootstrap_and_shared_image_root_copy(self):
        flags,models,conditions,roots=fixture(6)
        # Two questions per image, different failure/eligibility allocations.
        flags[0,0,41:60]=8
        flags[0,1,41:45]=8
        flags[1,4,1]=8
        flags[2,2,90]=8
        result=self.function('analyze')(flags,models,conditions,roots,bootstrap=15,seed=17)
        observed={(r['view'],r['model'],r['metric'],r['draw_id']):r['value'] for r in result['tables']['bootstrap_values']}
        for view,qbit,obit in jm.VIEWS:
            q=flags&qbit != 0
            e=(q[:,:,ANCHORS[:,0]] & q[:,:,ANCHORS[:,1]]).all(axis=0)
            s=q[:,:,:41].all(axis=2).all(axis=0)
            valid=e.sum(axis=0)>=30
            for mi,model in enumerate(models):
                outcome=flags[mi,:,41:]&obit != 0
                for di,draw in enumerate(result['draws']):
                    # Independent root expansion: all questions copied together.
                    ix=np.concatenate([np.flatnonzero(np.array([r['image_fingerprint'] for r in roots])==str(image)) for image in draw])
                    expected_jsr=float(outcome[ix][e[ix]].mean()) if e[ix].any() else None
                    expected_jss=float(outcome[ix][s[ix]].all(axis=1).mean()) if s[ix].any() else None
                    self.assertAlmostEqual(observed[view,model,'JSR',di],expected_jsr)
                    self.assertAlmostEqual(observed[view,model,'JSS',di],expected_jss)
                    self.assertIsNone(observed[view,model,'TGR5',di])

    def test_candidate_integer_inclusion_and_fixed_lexical_selection(self):
        flags,models,conditions,roots=fixture()
        result=self.function('analyze')(flags,models,conditions,roots,bootstrap=0)
        selected=[r for r in result['tables']['matched_selected'] if r['view']=='Task']
        self.assertEqual([r['selection_rank'] for r in selected],[1,2,3])
        self.assertTrue(all(r['model']=='a' and r['exact_equal_correct'] for r in selected))
        tuples=[(r['model'],r['pair_a'],r['pair_b']) for r in selected]
        allkeys=sorted((r['model'],r['pair_a'],r['pair_b']) for r in result['tables']['matched_candidates'] if r['view']=='Task')
        self.assertEqual(tuples,allkeys[:3])

    def test_cli_authentication_fresh_output_and_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp)
            # No data: a started receipt followed by an explicit preserved failure.
            with self.assertRaises(FileNotFoundError):
                jm.run(temp/'missing',temp/'attempt',0,17,BASE/'frozen_protocol_v2.md')
            self.assertTrue((temp/'attempt'/'run_manifest.json').is_file())
            self.assertTrue((temp/'attempt'/'failure.json').is_file())
            with self.assertRaises(FileExistsError):
                jm.run(temp/'missing',temp/'attempt',0,17,BASE/'frozen_protocol_v2.md')
            wrong=temp/'wrong.md'
            wrong.write_text('wrong protocol',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'protocol SHA256'):
                jm.run(temp/'missing',temp/'no_run',0,17,wrong)
            self.assertFalse((temp/'no_run').exists())

if __name__=='__main__':
    unittest.main(verbosity=2)
