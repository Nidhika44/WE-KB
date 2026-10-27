"""Export simulation records, convergence traces, and run metadata."""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class DataExporter:

    def __init__(self, base_output_dir: str = "output"):
        self.base_dir = Path(base_output_dir)
        self.data_dir = self.base_dir / "data"
        self.figures_dir = self.base_dir / "figures"
        self._create_directories()

    def _create_directories(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def export_dataframe(
        self,
        df: pd.DataFrame,
        filename_prefix: str,
        export_parquet: bool = False,
        export_csv: bool = True,
    ) -> Dict[str, str]:
        saved_paths = {}

        if export_csv:
            csv_path = self.data_dir / f"{filename_prefix}.csv"
            df.to_csv(csv_path, index=False)
            saved_paths["csv"] = str(csv_path)

        if export_parquet:
            parquet_path = self.data_dir / f"{filename_prefix}.parquet"
            df.to_parquet(parquet_path, index=False)
            saved_paths["parquet"] = str(parquet_path)

        return saved_paths

    def export_convergence_traces(
        self,
        convergence_data: Dict[str, List[List[float]]],
        filename_prefix: str,
    ) -> str:
        json_path = self.data_dir / f"{filename_prefix}_convergence.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(convergence_data, f, indent=2)


        rows = []
        for algo, curves in convergence_data.items():
            if not curves:
                continue
            arr = np.array(curves)
            mean_curve = np.mean(arr, axis=0)
            std_curve = np.std(arr, axis=0)
            for it, (m_val, s_val) in enumerate(zip(mean_curve, std_curve)):
                rows.append({
                    "algorithm": algo,
                    "iteration": it,
                    "mean_fitness": m_val,
                    "std_fitness": s_val,
                })

        df_conv = pd.DataFrame(rows)
        csv_path = self.data_dir / f"{filename_prefix}_convergence.csv"
        df_conv.to_csv(csv_path, index=False)

        return str(json_path)

    def export_run_metadata(
        self, metadata: Dict[str, Any], filename_prefix: str
    ) -> str:
        meta_path = self.data_dir / f"{filename_prefix}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        return str(meta_path)
