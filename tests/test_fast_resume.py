import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import resume_research_fast as fast
from experiments.run_research import Scenario,run_case
from we_kb_fpa.checkpoint import implementation_signature,identity_hash,RunLock,Checkpoint


class TestFastResume(unittest.TestCase):
    def setup_run(self,folder,slots=2):
        output=Path(folder)
        scenarios=[Scenario('channel','one',users=8,edges=3,q=0),Scenario('channel','two',users=8,edges=3,q=0)]
        settings=dict(protocol='legacy-greedy-seeded-research-v1',seeds=[50001],scenarios=[asdict(s) for s in scenarios],
            evaluations=4,population=2,slots=slots,shock_slots=4,include_drl=False,checkpoint_every=1,
            reference_seconds=.1,mip_gap=.01,hardware=dict(processor=platform.processor(),
                threads={name:os.environ.get(name) for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']}))
        protocol=dict(implementation=implementation_signature(),settings=settings)
        for name in ['config.json','frozen_protocol.json']:
            (output/name).write_text(json.dumps(protocol))
        root=output/'checkpoints'/identity_hash(settings)[:24];root.mkdir(parents=True)
        return output,root,settings,scenarios

    def test_cached_cases_are_not_recomputed_or_reexported(self):
        with tempfile.TemporaryDirectory() as folder:
            output,root,settings,scenarios=self.setup_run(folder)
            rows=run_case(settings,root,scenarios[0],50001,42)
            directory=output/'per_seed';directory.mkdir()
            path=directory/'channel-one-50001.csv';pd.DataFrame(rows).to_csv(path,index=False)
            original=path.read_bytes();timestamp=path.stat().st_mtime_ns
            frozen=(output/'frozen_protocol.json').read_bytes()
            with patch.object(fast,'run_case',wraps=run_case) as compute,patch.object(fast,'report'),patch.object(fast,'write_aggregate',wraps=fast.write_aggregate) as write:
                fast.run(output)
                self.assertEqual(compute.call_count,1)
                self.assertEqual(write.call_count,2)
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(path.stat().st_mtime_ns,timestamp)
            self.assertEqual((output/'frozen_protocol.json').read_bytes(),frozen)
            frame=pd.read_csv(output/'per_seed_slots.csv')
            self.assertFalse(frame.duplicated(['seed','scenario','slot','algorithm']).any())
            self.assertEqual(len(frame),28)
            with patch.object(fast,'run_case') as compute,patch.object(fast,'report'):
                fast.run(output)
                compute.assert_not_called()

    def test_original_partial_slot_checkpoint_is_resumed(self):
        with tempfile.TemporaryDirectory() as folder:
            output,root,settings,scenarios=self.setup_run(folder,4)
            save=Checkpoint.save;captured=[]
            def interrupt(checkpoint,state,compact=False):
                save(checkpoint,state,compact)
                if state.get('next_slot')==2:
                    captured.extend(copy.deepcopy(state['records']));raise KeyboardInterrupt
            with patch.object(Checkpoint,'save',interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_case(settings,root,scenarios[0],50001,42)
            with patch.object(fast,'report') as report:
                fast.run(output)
            records=report.call_args.args[0].to_dict('records')
            for expected,actual in zip(captured,records):
                self.assertEqual(expected['search_objective'],actual['search_objective'])
                self.assertEqual(expected['runtime_ms'],actual['runtime_ms'])
            self.assertEqual(len(records),56)

    def test_active_writer_is_not_interrupted(self):
        with tempfile.TemporaryDirectory() as folder:
            output,*_=self.setup_run(folder)
            with RunLock(output/'run.lock'):
                with self.assertRaisesRegex(RuntimeError,'still active'):
                    fast.run(output)

    def test_frozen_signature_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            output,*_=self.setup_run(folder)
            for name in ['frozen_protocol.json','config.json']:
                value=json.loads((output/name).read_text());value['implementation']='mismatch'
                (output/name).write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'code or dependency'):
                fast.read_protocol(output)


if __name__=='__main__':
    unittest.main()
