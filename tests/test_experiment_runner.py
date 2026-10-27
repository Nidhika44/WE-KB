import unittest
import pandas as pd
from we_kb_fpa.experiment_runner import ExperimentRunner
from we_kb_fpa.publication_plots import PublicationPlotter


class TestExperimentRunner(unittest.TestCase):
    def test_parallel_reproducibility(self):
        def configure(value, net, channel, task, cost, opt):
            net.num_users = 4
            net.num_edge_servers = 2
            opt.population_size = 6
            opt.max_iterations = 2

        kwargs = dict(scenario_name="test", param_values=[1.], param_updater_fn=configure,
                      algorithms=["Random", "WE-KB-FPA"], num_trials=2)
        serial, curves = ExperimentRunner(num_workers=1).run_sweep(**kwargs)
        parallel, other_curves = ExperimentRunner(num_workers=2).run_sweep(**kwargs)
        pd.testing.assert_frame_equal(serial.drop(columns="runtime_ms"), parallel.drop(columns="runtime_ms"))
        self.assertEqual(curves, other_curves)

    def test_small_sample_interval(self):
        self.assertGreater(PublicationPlotter._compute_ci95(pd.Series([1., 2., 3.])), 1.96 / 3 ** .5)
        self.assertEqual(PublicationPlotter._compute_ci95(pd.Series([1.])), 0.)
