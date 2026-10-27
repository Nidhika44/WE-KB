from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import pandas as pd
import research_parallel as parallel
from research_serial_benchmark import samples,timing_case
from experiments.run_research import Scenario,run_case
from we_kb_fpa.checkpoint import identity_hash,implementation_signature,RunLock
from we_kb_fpa.evaluation import fingerprint


class TestParallelResearch(unittest.TestCase):
    def setup_run(self,folder):
        base=Path(folder)
        scenarios=[Scenario('scalability','one',users=8,edges=3,q=0),Scenario('csi','two',users=8,edges=3,q=0,csi_error_db=3.)]
        settings=dict(protocol='legacy-greedy-seeded-research-v1',seeds=[50001],scenarios=[asdict(s) for s in scenarios],
            evaluations=12,population=4,slots=4,shock_slots=4,include_drl=False,checkpoint_every=1,
            reference_seconds=.1,mip_gap=.01,hardware=dict(processor=platform.processor(),
                threads={name:os.environ.get(name) for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']}))
        frozen=dict(settings=settings,implementation=implementation_signature())
        for name in ['config.json','frozen_protocol.json']:
            (base/name).write_text(json.dumps(frozen))
        root=base/'checkpoints'/identity_hash(settings)[:24];root.mkdir(parents=True)
        return base,root,settings,scenarios

    def test_time_sensitive_metrics_are_never_pooled(self):
        row=dict(search_objective=12.,violation_pct=25.,runtime_ms=3.,retrieval_ms=.1,
                 delay_adjusted_violation_pct=50.,delay_feasible_cost=4.,kb_build_seconds=2.)
        frame=parallel.quality_frame([row],'parallel')
        self.assertEqual(frame.search_objective.iloc[0],12.)
        self.assertEqual(frame.violation_pct.iloc[0],25.)
        for field in parallel.TIMING_FIELDS:
            self.assertTrue(frame[field].isna().all())

    def test_parallel_preserves_completed_case_and_matches_serial_quality(self):
        with tempfile.TemporaryDirectory() as folder,tempfile.TemporaryDirectory() as reference:
            base,root,settings,scenarios=self.setup_run(folder)
            first=run_case(settings,root,scenarios[0],50001,42)
            cached=parallel.checkpoint_for(settings,root,scenarios[0],50001).path
            before=hashlib.sha256(cached.read_bytes()).hexdigest()
            expected=run_case(settings,Path(reference),scenarios[1],50001,42)
            with patch.object(parallel,'produce_quality_report') as reporting:
                parallel.run(base,workers=2,skip_serial=True)
            frame=pd.concat(reporting.call_args.args[0],ignore_index=True)
            self.assertEqual(len(frame),len(first)+len(expected))
            rows=frame[frame.scenario=='two'].sort_values(['slot','algorithm'])
            target=pd.DataFrame(expected).sort_values(['slot','algorithm'])
            for metric in ['search_objective','feasible_cost','violation_pct','physical_violation_lower_bound_pct']:
                np.testing.assert_allclose(rows[metric].to_numpy(),target[metric].to_numpy(),rtol=1e-12,atol=1e-12)
            self.assertTrue(frame.runtime_ms.isna().all())
            self.assertEqual(hashlib.sha256(cached.read_bytes()).hexdigest(),before)
            self.assertEqual(json.loads((base/'parallel_option/quality_complete.json').read_text())['computed'],1)
            with patch.object(parallel,'produce_quality_report'):
                parallel.run(base,workers=2,skip_serial=True)
            self.assertEqual(json.loads((base/'parallel_option/quality_complete.json').read_text())['computed'],0)

    def test_serial_snapshots_reproduce_physical_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            base,root,settings,scenarios=self.setup_run(folder)
            scenario=scenarios[1]
            rows=run_case(settings,root,scenario,50001,42)
            for slot,phase,args,realized,net in samples(settings,scenario,50001):
                expected=next(row for row in rows if row['slot']==slot)
                self.assertEqual(fingerprint(args),expected['observation_hash'])
                self.assertEqual(fingerprint((*args[:4],realized,args[5])),expected['realization_hash'])
            measured=timing_case(settings,base/'serial',scenario,50001,42)
            self.assertEqual({row['slot'] for row in measured},{0,2})
            self.assertTrue(all(row['runtime_ms']>0 for row in measured))
            self.assertTrue(all(row['benchmark_mode']=='isolated_serial_fresh_controller_snapshots' for row in measured))

    def test_export_contains_no_oci_credentials_and_validates(self):
        with tempfile.TemporaryDirectory() as folder,tempfile.TemporaryDirectory() as bundle:
            base,root,settings,scenarios=self.setup_run(folder)
            run_case(settings,root,scenarios[0],50001,42)
            archive=Path(bundle)/'transfer.zip'
            transfer=parallel.export_quality(base,archive)
            self.assertEqual(len(transfer['completed_cases']),1)
            with zipfile.ZipFile(archive) as z:
                self.assertFalse(any(name.endswith(('.pem','.pub')) or 'id_rsa' in name for name in z.namelist()))
                z.extractall(bundle)
            config,_,manifest=parallel.context(Path(bundle)/'transfer',portable=True)
            self.assertEqual(config['evaluations'],settings['evaluations'])
            self.assertEqual(manifest['source_sha256'],transfer['source_sha256'])

    def test_original_active_writer_is_not_interrupted(self):
        with tempfile.TemporaryDirectory() as folder:
            base,*_=self.setup_run(folder)
            with RunLock(base/'run.lock'):
                with self.assertRaises(RuntimeError):
                    parallel.run(base,workers=2,skip_serial=True)


if __name__=='__main__':
    unittest.main()
