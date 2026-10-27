"""Compare algorithms across input task sizes."""

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


def run_workload_benchmark(
    d_k_mb_values: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0],
    num_trials: int = 15,
    num_workers: int = 1,
    resume: bool = True,
) -> pd.DataFrame:
    runner = ExperimentRunner(base_seed=303, num_workers=num_workers, checkpoint_dir="output/checkpoints/static", resume=resume)
    algorithms = ["Greedy-Latency", "GA", "PSO", "FPA-Baseline", "WE-FPA", "WE-KB-FPA"]

    def update_workload(val, net_cfg, ch_cfg, task_cfg, cost_cfg, opt_cfg):

        task_cfg.d_k_min_mb = max(0.5, val * 0.8)
        task_cfg.d_k_max_mb = val * 1.2

    df, conv_map = runner.run_sweep(
        scenario_name="Workload_TaskSize_Dk",
        param_values=d_k_mb_values,
        param_updater_fn=update_workload,
        algorithms=algorithms,
        num_trials=num_trials,
        run_exact=False,
    )

    exporter = DataExporter()
    exporter.export_dataframe(df, "workload_qos_benchmark")

    plotter = PublicationPlotter()
    plotter.plot_fig3_qos_compliance(df)
    print("Workload QoS experiment complete. Saved to output/figures/fig3_qos_compliance.pdf")
    return df


if __name__ == "__main__":
    run_workload_benchmark(num_trials=5)
