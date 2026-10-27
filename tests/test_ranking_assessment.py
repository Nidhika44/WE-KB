from dataclasses import asdict,replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from experiments.assess_ranking import matched_configs,report
from we_kb_fpa.anytime import AnytimeConfig


class TestRankingAssessment(unittest.TestCase):
    def test_only_ranking_can_differ_in_matched_search(self):
        fpa=AnytimeConfig(strategy='fpa',event_trigger=False)
        ga=replace(fpa,strategy='ga')
        settings={'No-Event-10ms':asdict(fpa),'KB-Categorical-GA-10ms':asdict(ga),
                  'DelayRank-FPA-10ms':asdict(replace(fpa,deadline_aware=True,stopping_enabled=False)),
                  'DelayRank-GA-10ms':asdict(replace(ga,deadline_aware=True,stopping_enabled=False)),
                  'Anchor-Only-10ms':asdict(replace(fpa,strategy='anchor'))}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'configs.json'
            path.write_text(json.dumps(settings))
            self.assertEqual(len(matched_configs(path,10.)),5)
            settings['DelayRank-GA-10ms']['population']+=1
            path.write_text(json.dumps(settings))
            with self.assertRaises(ValueError):
                matched_configs(path,10.)

    def test_development_gate_does_not_pass_when_only_one_strategy_improves(self):
        rows=[]
        costs={'No-Event-10ms':100.,'KB-Categorical-GA-10ms':100.,'DelayRank-FPA-10ms':98.,
               'DelayRank-GA-10ms':100.,'Greedy-Uniform':110.,'Anchor-Only-10ms':105.}
        for seed in range(10011,10021):
            for fade in [0.,15.,35.]:
                for algorithm,cost in costs.items():
                    rows.append(dict(seed=seed,fade=fade,phase='post',algorithm=algorithm,cost=cost,
                                     delay_adjusted_cost=cost,delay_adjusted_violation_pct=10.,runtime_ms=1.))
        with tempfile.TemporaryDirectory() as folder,patch('sys.stdout',new=io.StringIO()):
            output=Path(folder)
            report(pd.DataFrame(rows),output,1.)
            result=json.loads((output/'assessment.json').read_text())
            self.assertTrue(result['strategies']['FPA']['passes_development_screen'])
            self.assertFalse(result['strategies']['GA']['passes_development_screen'])
            self.assertFalse(result['final_evaluation_run'])
            self.assertIn('did not pass',result['recommendation'])


if __name__=='__main__':
    unittest.main()
