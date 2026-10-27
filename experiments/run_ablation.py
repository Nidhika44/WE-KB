"""Evaluate component ablations and convergence."""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import List, Dict
import numpy as np
import pandas as pd

from we_kb_fpa.config import NetworkConfig, ChannelConfig, TaskConfig, CostConfig, OptimizerConfig
from we_kb_fpa.experiment_runner import ExperimentRunner
from we_kb_fpa.data_exporter import DataExporter
from we_kb_fpa.publication_plots import PublicationPlotter


def run_ablation_benchmark(num_trials: int = 25, num_workers: int = 1, resume: bool = True) -> pd.DataFrame:
    runner = ExperimentRunner(base_seed=404, num_workers=num_workers, checkpoint_dir="output/checkpoints/static", resume=resume)
    ablation_algorithms = [
        "FPA-Baseline",
        "KB-FPA",
        "WE-FPA",
        "WE-KB-FPA",
        "PSO",
        "GA",
    ]

    def identity_updater(val, net_cfg, ch_cfg, task_cfg, cost_cfg, opt_cfg):
        pass

    df, conv_map = runner.run_sweep(
        scenario_name="Formal_Ablation_Study",
        param_values=[1.0],
        param_updater_fn=identity_updater,
        algorithms=ablation_algorithms,
        num_trials=num_trials,
        run_exact=False,
    )

    exporter = DataExporter()
    exporter.export_dataframe(df, "formal_ablation_benchmark")
    exporter.export_convergence_traces(conv_map, "ablation_convergence")

    plotter = PublicationPlotter()
    plotter.plot_fig4_convergence(conv_map)
    plotter.plot_fig5_ablation(df)
    print("Ablation benchmark complete. Saved to output/figures/fig4_convergence_curves.pdf and fig5_ablation_study.pdf")
    return df


if __name__ == "__main__":
    run_ablation_benchmark(num_trials=10)
