import copy
from dataclasses import asdict
import hashlib
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.run_topology_shock import tune_dqn, training_config, evaluate_checkpoint
from we_kb_fpa.checkpoint import Checkpoint, RunLock
from we_kb_fpa.dqn import DQNBaseline, DQNConfig
from we_kb_fpa.droo import DROOAdapted
from we_kb_fpa.evaluation import snapshots, fingerprint
from we_kb_fpa.experiment_runner import ExperimentRunner
from we_kb_fpa.training import train_seed


def model_digest(agent):
    digest = hashlib.sha256()
    for weights in (agent.weights, getattr(agent, "target", []), agent.adam.first or [], agent.adam.second or []):
        for array in weights:
            digest.update(array.tobytes())
    digest.update(repr((agent.updates, agent.adam.steps, getattr(agent, "epsilon", None), agent.rng.bit_generator.state)).encode())
    for transition in agent.memory:
        for value in transition:
            digest.update(value.tobytes() if isinstance(value, np.ndarray) else repr(value).encode())
    return digest.hexdigest()


def stub_evaluate(dqn, droo, quality, seed, budget, *unused, **kwargs):
    return [dict(seed=seed, training_budget=budget, dqn=model_digest(dqn), droo=model_digest(droo))]


class TestCheckpoint(unittest.TestCase):
    def test_atomic_backup_checksum_and_settings_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Checkpoint(folder, "state", dict(seed=1))
            checkpoint.save(dict(step=1))
            checkpoint.save(dict(step=2))
            self.assertEqual(checkpoint.load()["step"], 2)
            checkpoint.path.write_bytes(b"partial write")
            self.assertEqual(checkpoint.load()["step"], 1)
            checkpoint.save(dict(step=3))
            checkpoint.path.write_bytes(b"another partial write")
            self.assertEqual(checkpoint.load()["step"], 1)
            self.assertIsNone(Checkpoint(folder, "state", dict(seed=2)).load())
            self.assertIsNone(Checkpoint(folder, "state", dict(seed=1), resume=False).load())
            checkpoint.previous.write_bytes(b"corrupted too")
            with self.assertRaises(RuntimeError):
                checkpoint.load()

    def test_output_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.lock"
            with RunLock(path):
                with self.assertRaises(RuntimeError):
                    with RunLock(path):
                        pass
            with RunLock(path):
                pass

    def test_simulation_stream_restores_random_and_mobility_state(self):
        stream = snapshots(42, 8, fading_db=35., midpoint=4)
        for _ in range(3):
            next(stream)
        restored = pickle.loads(pickle.dumps(stream))
        for original, replayed in zip(stream, restored):
            self.assertEqual(original[0], replayed[0])
            self.assertEqual(fingerprint(original[1]), fingerprint(replayed[1]))
            self.assertEqual(original[2:], replayed[2:])

    def test_static_trial_resume_keeps_convergence_and_skips_completed(self):
        def configure(*args):
            pass
        kwargs = dict(scenario_name="resume", param_values=[1.], param_updater_fn=configure,
                      algorithms=["Random"], num_trials=3)
        with tempfile.TemporaryDirectory() as folder:
            runner = ExperimentRunner(checkpoint_dir=folder)
            def interrupted(**args):
                if args["slot_idx"] == 1:
                    raise KeyboardInterrupt()
                return [dict(algorithm="Random", cost_j=args["slot_idx"], convergence_curve=[3., 2.])]
            with patch("we_kb_fpa.experiment_runner._run_single_trial_worker", side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    runner.run_sweep(**kwargs)
            calls = []
            def remaining(**args):
                calls.append(args["slot_idx"])
                return [dict(algorithm="Random", cost_j=args["slot_idx"], convergence_curve=[3., 2.])]
            with patch("we_kb_fpa.experiment_runner._run_single_trial_worker", side_effect=remaining):
                df, curves = runner.run_sweep(**kwargs)
            self.assertEqual(calls, [1, 2])
            self.assertEqual(len(df), 3)
            self.assertEqual(curves["Random"], [[3., 2.]]*3)
            with patch("we_kb_fpa.experiment_runner._run_single_trial_worker", side_effect=AssertionError("must use cache")):
                again, other = runner.run_sweep(**kwargs)
            pd.testing.assert_frame_equal(df, again)
            self.assertEqual(curves, other)

    def test_tuning_resumes_inside_candidate_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = DQNBaseline.run
            count = 0
            def interrupt(agent, *args, **kwargs):
                nonlocal count
                out = original(agent, *args, **kwargs)
                if kwargs.get("training"):
                    count += 1
                    if count == 5:
                        raise KeyboardInterrupt()
                return out
            with patch.object(DQNBaseline, "run", interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    tune_dqn(8, 2, root, root/"resume", checkpoint_every=2)
            count = 0
            def track(agent, *args, **kwargs):
                nonlocal count
                if kwargs.get("training"):
                    count += 1
                return original(agent, *args, **kwargs)
            with patch.object(DQNBaseline, "run", track):
                resumed = tune_dqn(8, 2, root, root/"resume", checkpoint_every=2)
            self.assertEqual(count, 44)
            results = pd.read_csv(root / "hyperparameter_validation.csv")
            baseline = tune_dqn(8, 2, root, root/"baseline", checkpoint_every=2)
            pd.testing.assert_frame_equal(results, pd.read_csv(root/"hyperparameter_validation.csv"))
            self.assertEqual(asdict(resumed), asdict(baseline))

    def test_training_resume_restores_replay_adam_rng_and_truncates_log(self):
        config = dict(training_slots=12, validation_slots=2, validation_interval=6,
                      training_budgets=[6, 12], slots=6, fading_values=[35.], population=4,
                      iterations=1, require_nominal_quality=False)
        selected = training_config(DQNConfig(batch_size=4, target_interval=2), 12)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, baseline_log = train_seed(42, config, selected, root/"baseline", root, stub_evaluate, checkpoint_every=2)
            original = DROOAdapted.run
            count = 0
            def interrupted(agent, *args, **kwargs):
                nonlocal count
                if kwargs.get("training"):
                    count += 1
                    if count == 5:
                        raise KeyboardInterrupt()
                return original(agent, *args, **kwargs)
            with patch.object(DROOAdapted, "run", interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    train_seed(42, config, selected, root/"resume", root, stub_evaluate, checkpoint_every=2)
            log = next((root/"resume").glob("*.csv"))
            with log.open("a") as stream:
                stream.write("uncommitted partial CSV row\n")
            resumed, resumed_log = train_seed(42, config, selected, root/"resume", root, stub_evaluate, checkpoint_every=2)
            self.assertEqual(baseline["records"], resumed["records"])
            self.assertEqual(baseline["validations"], resumed["validations"])
            pd.testing.assert_frame_equal(pd.read_csv(baseline_log), pd.read_csv(resumed_log))
            self.assertEqual(len(pd.read_csv(resumed_log)), 24)
            with patch.object(DQNBaseline, "run", side_effect=AssertionError("completed seed must not retrain")):
                cached, _ = train_seed(42, config, selected, root/"resume", root, stub_evaluate)
            self.assertEqual(cached["records"], resumed["records"])

    def test_evaluation_resume_preserves_online_learning_and_knowledge(self):
        dqn, droo = DQNBaseline(seed=42), DROOAdapted(seed=42, batch_size=2)
        for _, args, _, _ in snapshots(100042, 4):
            dqn.run(*args, training=True)
            droo.run(*args, training=True)
        quality = {"dqn": dict(nominal_quality_pass=False), "droo": dict(nominal_quality_pass=False)}
        from we_kb_fpa.we_kb_fpa import WEKBFPAOptimizer
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = dqn, droo, quality, 42, 4, 6, [35.], 4, 1
            baseline = evaluate_checkpoint(*args, checkpoint_root=root/"baseline", checkpoint_every=2)
            original, count = WEKBFPAOptimizer.optimize, 0
            def interrupted(optimizer, *a, **kw):
                nonlocal count
                count += 1
                if count == 7:
                    raise KeyboardInterrupt()
                return original(optimizer, *a, **kw)
            with patch.object(WEKBFPAOptimizer, "optimize", interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    evaluate_checkpoint(*args, checkpoint_root=root/"resume", checkpoint_every=2)
            resumed = evaluate_checkpoint(*args, checkpoint_root=root/"resume", checkpoint_every=2)
            exclude = ["runtime_ms", "decision_ms", "learning_ms", "exceeds_slot", "delay_adjusted_cost", "delay_adjusted_latency_ms", "delay_adjusted_violation_pct"]
            pd.testing.assert_frame_equal(pd.DataFrame(baseline).drop(columns=exclude), pd.DataFrame(resumed).drop(columns=exclude))
            self.assertEqual(len(resumed), 72)
