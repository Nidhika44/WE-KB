"""Compare small-instance costs and solver runtimes."""

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


def run_optimality_gap_benchmark(
    n_values: List[int] = [3, 4, 5, 6, 7, 8],
    num_trials: int = 5,
    num_workers: int = 1,
    resume: bool = True,
) -> pd.DataFrame:
    runner = ExperimentRunner(base_seed=505, num_workers=num_workers, checkpoint_dir="output/checkpoints/static", resume=resume)
    algorithms = ["WE-KB-FPA", "PSO", "GA"]

    def update_small_n(val, net_cfg, ch_cfg, task_cfg, cost_cfg, opt_cfg):
        net_cfg.num_users = int(val)
        net_cfg.num_edge_servers = 2
        opt_cfg.population_size = 25
        opt_cfg.max_iterations = 40

    df, conv_map = runner.run_sweep(
        scenario_name="Uniform_Allocation_Reference",
        param_values=[float(n) for n in n_values],
        param_updater_fn=update_small_n,
        algorithms=algorithms,
        num_trials=num_trials,
        run_exact=True,
    )

    exporter = DataExporter()
    exporter.export_dataframe(df, "uniform_reference_benchmark")

    plotter = PublicationPlotter()
    plotter.plot_fig6_runtime_vs_exact(df)


    summary = df[df["algorithm"] != "Uniform-assignment enumeration"].groupby(["parameter_value", "algorithm"])["reference_difference_pct"].agg(["mean", "std"]).reset_index()
    print("\nSIGNED DIFFERENCE TO UNIFORM-ALLOCATION REFERENCE (%); NOT A JOINT OPTIMALITY GAP")
    print(summary.to_string(index=False))
    print("=================================================================\n")
    print("Reference experiment complete. Saved to output/figures/fig6_runtime_vs_reference.pdf")
    return df


if __name__ == "__main__":
    run_optimality_gap_benchmark(n_values=[3, 4, 5, 6], num_trials=3)
