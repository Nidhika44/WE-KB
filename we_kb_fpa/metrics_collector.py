"""Aggregate per-slot simulation metrics."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class AggregatedMetric:
    mean: float
    std: float
    ci_95_lower: float
    ci_95_upper: float


class MetricsCollector:

    def __init__(self):
        self.raw_records: List[Dict[str, Any]] = []

    def record_run(
        self,
        scenario_name: str,
        run_id: int,
        slot_id: int,
        algorithm: str,
        cost_j: float,
        latency_sec: float,
        latency_tx_sec: float,
        latency_queue_sec: float,
        latency_comp_sec: float,
        latency_prop_sec: float,
        latency_dl_sec: float,
        energy_joules: float,
        energy_comm_joules: float,
        energy_comp_joules: float,
        violation_rate_pct: float,
        jains_fairness: float,
        runtime_ms: float,
        iter_to_95: int,
        exact_cost: Optional[float] = None,
        exact_runtime_ms: Optional[float] = None,
        parameter_value: Optional[float] = None,
    ):
        opt_gap = None
        if exact_cost is not None and exact_cost > 0:
            opt_gap = max(0.0, ((cost_j - exact_cost) / exact_cost) * 100.0)

        record = {
            "scenario": scenario_name,
            "run_id": run_id,
            "slot_id": slot_id,
            "algorithm": algorithm,
            "cost_j": cost_j,
            "latency_ms": latency_sec * 1000.0,
            "latency_tx_ms": latency_tx_sec * 1000.0,
            "latency_queue_ms": latency_queue_sec * 1000.0,
            "latency_comp_ms": latency_comp_sec * 1000.0,
            "latency_prop_ms": latency_prop_sec * 1000.0,
            "latency_dl_ms": latency_dl_sec * 1000.0,
            "energy_j": energy_joules,
            "energy_comm_j": energy_comm_joules,
            "energy_comp_j": energy_comp_joules,
            "violation_rate_pct": violation_rate_pct,
            "jains_fairness": jains_fairness,
            "runtime_ms": runtime_ms,
            "iter_to_95": iter_to_95,
            "exact_cost": exact_cost,
            "exact_runtime_ms": exact_runtime_ms,
            "optimality_gap_pct": opt_gap,
            "parameter_value": parameter_value,
        }
        self.raw_records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.raw_records)

    def compute_summary_table(
        self, group_by_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()

        groups = group_by_cols or ["scenario", "algorithm"]
        metrics = [
            "cost_j",
            "latency_ms",
            "energy_j",
            "violation_rate_pct",
            "jains_fairness",
            "runtime_ms",
            "iter_to_95",
        ]
        if "optimality_gap_pct" in df and df["optimality_gap_pct"].notnull().any():
            metrics.append("optimality_gap_pct")

        agg_dict = {m: ["mean", "std"] for m in metrics if m in df.columns}
        summary_df = df.groupby(groups).agg(agg_dict).reset_index()
        return summary_df

    @staticmethod
    def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> AggregatedMetric:
        arr = np.asarray(data)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

        if n <= 1:
            return AggregatedMetric(mean, std, mean, mean)


        margin = 1.96 * (std / np.sqrt(n))
        return AggregatedMetric(
            mean=mean,
            std=std,
            ci_95_lower=mean - margin,
            ci_95_upper=mean + margin,
        )
