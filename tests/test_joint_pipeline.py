import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.run_joint import main
from experiments.train_joint import load_model
from we_kb_fpa.checkpoint import Checkpoint


class TestJointPipeline(unittest.TestCase):
    def test_model_artifact_is_bound_to_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Checkpoint(folder,'model',{'seed':42})
            checkpoint.save(dict(complete=True,best='trained-state'))
            manifest=dict(models={'dqn-42':dict(root=folder,name='model',metadata={'seed':42},
                sha256=hashlib.sha256(checkpoint.path.read_bytes()).hexdigest())})
            self.assertEqual(load_model(manifest,'dqn',42),'trained-state')
            checkpoint.save(dict(complete=True,best='replacement-state'))
            with self.assertRaisesRegex(ValueError,'changed'):
                load_model(manifest,'dqn',42)

    def test_smoke_models_cannot_access_final_trajectories(self):
        with tempfile.TemporaryDirectory() as folder:
            argv=['run_joint','--role','final','--models-dir','unused','--output-dir',folder,'--freeze-protocol']
            with patch('sys.argv',argv),patch('experiments.run_joint.load_models',return_value={'status':'smoke_only'}),patch('experiments.run_joint.run_case') as run:
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    main()
                run.assert_not_called()
                self.assertFalse((Path(folder)/'frozen_protocol.json').exists())

    def test_final_requires_independent_model_replicates(self):
        with tempfile.TemporaryDirectory() as folder:
            argv=['run_joint','--role','final','--rigorous','--models-dir','unused','--output-dir',folder,'--freeze-protocol']
            with patch('sys.argv',argv),patch('experiments.run_joint.load_models',return_value={'status':'trained','training':{'seeds':[42]}}),patch('experiments.run_joint.run_case') as run:
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    main()
                run.assert_not_called()


if __name__=='__main__':
    unittest.main()
