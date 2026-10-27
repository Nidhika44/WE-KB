"""Compare algorithms across nominal radio power settings."""

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


def run_channel_degradation_benchmark(
    sinr_db_values: List[float] = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0],
    num_trials: int = 15,
    num_workers: int = 1,
    resume: bool = True,
) -> pd.DataFrame:
    runner = ExperimentRunner(base_seed=202, num_workers=num_workers, checkpoint_dir="output/checkpoints/static", resume=resume)
    algorithms = ["FPA-Baseline", "PSO", "GA", "WE-FPA", "WE-KB-FPA"]

    def update_sinr(val, net_cfg, ch_cfg, task_cfg, cost_cfg, opt_cfg):


        target_linear = 10.0 ** (val / 10.0)

        scale = max(0.01, target_linear / 10.0)
        net_cfg.p_tx_min = 0.1 * scale
        net_cfg.p_tx_max = 0.5 * scale

    df, conv_map = runner.run_sweep(
        scenario_name="Channel_Degradation_SINR",
        param_values=sinr_db_values,
        param_updater_fn=update_sinr,
        algorithms=algorithms,
        num_trials=num_trials,
        run_exact=False,
    )

    exporter = DataExporter()
    exporter.export_dataframe(df, "channel_resilience_benchmark")

    plotter = PublicationPlotter()
    plotter.plot_fig2_sinr_resilience(df)
    print("Wireless resilience experiment complete. Saved to output/figures/fig2_sinr_resilience.pdf")
    return df


if __name__ == "__main__":
    run_channel_degradation_benchmark(num_trials=5)
