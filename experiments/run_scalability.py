"""Compare algorithms across user counts."""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import List
import numpy as np
import pandas as pd

from we_kb_fpa.config import NetworkConfig, ChannelConfig, TaskConfig, CostConfig, OptimizerConfig
from we_kb_fpa.experiment_runner import ExperimentRunner
from we_kb_fpa.data_exporter import DataExporter
from we_kb_fpa.publication_plots import PublicationPlotter


def run_scalability_benchmark(
    n_values: List[int] = [10, 20, 50, 100, 200],
    num_trials: int = 15,
    num_workers: int = 1,
    resume: bool = True,
) -> pd.DataFrame:
    runner = ExperimentRunner(base_seed=101, num_workers=num_workers, checkpoint_dir="output/checkpoints/static", resume=resume)
    algorithms = [
        "Random",
        "Greedy-Latency",
        "Greedy-Energy",
        "GA",
        "PSO",
        "FPA-Baseline",
        "WE-KB-FPA",
    ]

    def update_n(val, net_cfg, ch_cfg, task_cfg, cost_cfg, opt_cfg):
        net_cfg.num_users = int(val)

        if val >= 100:
            opt_cfg.population_size = 40
            opt_cfg.max_iterations = 60

    df, conv_map = runner.run_sweep(
        scenario_name="Scalability_Users",
        param_values=[float(n) for n in n_values],
        param_updater_fn=update_n,
        algorithms=algorithms,
        num_trials=num_trials,
        run_exact=False,
    )

    exporter = DataExporter()
    exporter.export_dataframe(df, "scalability_benchmark")

    plotter = PublicationPlotter()
    plotter.plot_fig1_scalability(df)
    print("Scalability experiment complete. Saved to output/data/ and output/figures/fig1_scalability_cost.pdf")
    return df


if __name__ == "__main__":
    run_scalability_benchmark(n_values=[10, 20, 50, 100], num_trials=5)
