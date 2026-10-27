from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.run_research import Scenario,run_case
from quality_reports import create_quality_reports,timing_column
from quality_report_finalize import finalize


class TestQualityReporting(unittest.TestCase):
    def fixture(self,root):
        scenarios=[Scenario('deadline','tau-half',users=8,edges=3,q=0,tau_scale=.5,sinr_db=-5.),
                   Scenario('deadline','tau-one',users=8,edges=3,q=0,tau_scale=1.,sinr_db=-5.)]
        settings=dict(seeds=[50001],scenarios=[asdict(s) for s in scenarios],evaluations=4,population=2,
            slots=2,shock_slots=4,include_drl=False,checkpoint_every=1,reference_seconds=.1,mip_gap=.01)
        rows=[]
        for scenario in scenarios:
            rows.extend(run_case(settings,root/'checkpoints',scenario,50001,42))
        frame=pd.DataFrame(rows)
        for column in frame.columns:
            if timing_column(column):
                frame[column]=np.nan
        return frame,settings

    def test_blank_timings_do_not_create_false_crossovers_or_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);frame,settings=self.fixture(root)
            with patch('quality_reports.plot_sweep'):
                create_quality_reports(frame,root/'report',settings)
            summary=pd.read_csv(root/'report/summary_ci95.csv')
            self.assertFalse(any(timing_column(metric) for metric in summary.metric))
            data=pd.read_csv(root/'report/quality_slots.csv')
            self.assertFalse(any(timing_column(column) for column in data.columns))
            crossover=pd.read_csv(root/'report/deadline_crossover_brackets.csv')
            self.assertIn('Unavailable',crossover.status.iloc[0])
            self.assertTrue(np.isfinite(summary[summary.metric=='violation_pct']['mean']).all())

    def test_finalizer_requires_all_cases_and_preserves_quality_values(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);frame,settings=self.fixture(root)
            output=root/'parallel';(output/'quality_results').mkdir(parents=True)
            (output/'parallel_protocol.json').write_text(json.dumps(dict(settings=settings)))
            incomplete=frame[frame.scenario=='tau-half']
            incomplete.to_csv(output/'quality_results/quality_slots.csv',index=False)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                finalize(output)
            frame.to_csv(output/'quality_results/quality_slots.csv',index=False)
            expected=frame.search_objective.to_numpy()
            with patch('quality_reports.plot_sweep'):
                finalize(output)
            actual=pd.read_csv(output/'quality_results/quality_slots.csv').search_objective.to_numpy()
            np.testing.assert_allclose(actual,expected,rtol=1e-12)
            self.assertEqual(json.loads((output/'quality_complete.json').read_text())['total'],2)


if __name__=='__main__':
    unittest.main()
