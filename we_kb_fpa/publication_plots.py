"""Export benchmark figures as vector PDF and 300-DPI PNG."""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import t


def setup_ieee_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "axes.labelsize": 9,
        "font.size": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 10,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "axes.grid": True,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


ALGO_STYLES = {
    "WE-KB-FPA": {"color": "#B22222", "marker": "o", "label": "WE-KB-FPA (Proposed)", "ls": "-"},
    "WE-FPA": {"color": "#E65100", "marker": "s", "label": "WE-FPA", "ls": "--"},
    "KB-FPA": {"color": "#6A1B9A", "marker": "^", "label": "KB-FPA", "ls": "-."},
    "FPA-Baseline": {"color": "#1565C0", "marker": "v", "label": "FPA (Baseline)", "ls": ":"},
    "PSO": {"color": "#00838F", "marker": "d", "label": "PSO", "ls": "--"},
    "GA": {"color": "#2E7D32", "marker": "p", "label": "Genetic Algorithm (GA)", "ls": "-."},
    "Greedy-Latency": {"color": "#5D4037", "marker": "<", "label": "Greedy (Latency)", "ls": ":"},
    "Greedy-Energy": {"color": "#455A64", "marker": ">", "label": "Greedy (Energy)", "ls": ":"},
    "Random": {"color": "#9E9E9E", "marker": "x", "label": "Random", "ls": ":"},
    "Exact MINLP (Gurobi)": {"color": "#000000", "marker": "*", "label": "Exact MINLP (Gurobi)", "ls": "-"},
}


class PublicationPlotter:

    def __init__(self, output_dir: str = "output/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_ieee_style()

    def _save_figure(self, fig: plt.Figure, name_prefix: str):
        pdf_path = self.output_dir / f"{name_prefix}.pdf"
        png_path = self.output_dir / f"{name_prefix}.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    @staticmethod
    def _compute_ci95(data_series: pd.Series) -> pd.Series:
        """Student t interval for independent trial means; zero margin for one trial."""
        n = len(data_series)
        std = data_series.std(ddof=1) if n > 1 else 0.0
        return float(t.ppf(.975, n - 1) * std / np.sqrt(n)) if n > 1 else 0.0

    def plot_fig1_scalability(self, df_scalability: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(3.6, 3.2))

        algorithms = [
            "Random",
            "Greedy-Latency",
            "Greedy-Energy",
            "GA",
            "PSO",
            "FPA-Baseline",
            "WE-KB-FPA",
        ]

        for algo in algorithms:
            df_a = df_scalability[df_scalability["algorithm"] == algo]
            if df_a.empty:
                continue

            summary = df_a.groupby("parameter_value")["cost_j"].agg(
                mean="mean",
                ci=self._compute_ci95
            ).reset_index()
            style = ALGO_STYLES.get(algo, {"color": "black", "marker": "o", "label": algo, "ls": "-"})

            x_vals = summary["parameter_value"]
            y_mean = summary["mean"]
            y_ci = summary["ci"]

            ax.plot(
                x_vals,
                y_mean,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["ls"],
                lw=2.0 if "WE-KB-FPA" in algo else 1.2,
                ms=5 if "WE-KB-FPA" in algo else 3.5,
            )
            ax.fill_between(
                x_vals,
                np.maximum(1e-1, y_mean - y_ci),
                y_mean + y_ci,
                color=style["color"],
                alpha=0.15,
            )

        ax.set_yscale("log")
        ax.set_xlabel(r"Number of User Devices ($N$)")
        ax.set_ylabel(r"Total System Cost ($J$) [Log Scale]")
        ax.set_title("System Scalability vs. User Load", y=1.26, fontsize=10)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=True,
            fontsize=5.8,
            columnspacing=0.8,
            handletextpad=0.3,
            handlelength=1.5,
        )
        self._save_figure(fig, "fig1_scalability_cost")

    def plot_fig2_sinr_resilience(self, df_sinr: pd.DataFrame):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True)

        key_algos = ["FPA-Baseline", "PSO", "GA", "WE-FPA", "WE-KB-FPA"]

        for algo in key_algos:
            df_a = df_sinr[df_sinr["algorithm"] == algo]
            if df_a.empty:
                continue


            grp = df_a.groupby("parameter_value")
            lat_mean = grp["latency_ms"].mean()
            lat_ci = grp["latency_ms"].apply(self._compute_ci95)

            eng_mean = grp["energy_j"].mean()
            eng_ci = grp["energy_j"].apply(self._compute_ci95)

            style = ALGO_STYLES.get(algo, {"color": "black", "marker": "o", "label": algo, "ls": "-"})
            sinr_x = lat_mean.index.values


            ax1.plot(
                sinr_x,
                lat_mean.values,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["ls"],
                lw=1.8 if "WE-KB-FPA" in algo else 1.2,
            )
            ax1.fill_between(
                sinr_x,
                np.maximum(0.0, lat_mean.values - lat_ci.values),
                lat_mean.values + lat_ci.values,
                color=style["color"],
                alpha=0.15,
            )


            ax2.plot(
                sinr_x,
                eng_mean.values,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["ls"],
                lw=1.8 if "WE-KB-FPA" in algo else 1.2,
            )
            ax2.fill_between(
                sinr_x,
                np.maximum(0.0, eng_mean.values - eng_ci.values),
                eng_mean.values + eng_ci.values,
                color=style["color"],
                alpha=0.15,
            )

        ax1.set_xlabel("Mean Channel SINR (dB)")
        ax1.set_ylabel("Average Latency $\\bar{\\tau}$ (ms)")
        ax1.set_title("(a) Latency Resilience under Fading")
        ax1.set_ylim(bottom=0.0)
        ax1.legend(frameon=True, fontsize=6.5, loc="upper right")

        ax2.set_xlabel("Mean Channel SINR (dB)")
        ax2.set_ylabel("Total Energy $\\bar{E}$ (Joules)")
        ax2.set_title("(b) Energy Consumption under Fading")
        ax2.set_ylim(bottom=0.0)
        ax2.legend(frameon=True, fontsize=6.5, loc="upper right")

        self._save_figure(fig, "fig2_sinr_resilience")

    def plot_fig3_qos_compliance(self, df_workload: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))

        algos = ["Greedy-Latency", "GA", "PSO", "FPA-Baseline", "WE-FPA", "WE-KB-FPA"]

        for algo in algos:
            df_a = df_workload[df_workload["algorithm"] == algo]
            if df_a.empty:
                continue

            summary = df_a.groupby("parameter_value")["violation_rate_pct"].agg(
                mean="mean",
                ci=self._compute_ci95
            ).reset_index()
            style = ALGO_STYLES.get(algo, {"color": "black", "marker": "o", "label": algo, "ls": "-"})

            x_vals = summary["parameter_value"]
            y_mean = summary["mean"]
            y_ci = summary["ci"]

            ax.plot(
                x_vals,
                y_mean,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["ls"],
                lw=2.0 if "WE-KB-FPA" in algo else 1.2,
            )
            ax.fill_between(
                x_vals,
                np.maximum(0.0, y_mean - y_ci),
                np.minimum(100.0, y_mean + y_ci),
                color=style["color"],
                alpha=0.15,
            )

        ax.set_xlabel("Average Task Data Size $D_k$ (Megabits)")
        ax.set_ylabel("QoS Deadline Violation Rate $R_{viol}$ (%)")
        ax.set_title("QoS Deadline Compliance")
        ax.set_ylim(bottom=-2.0, top=105.0)
        ax.legend(frameon=True, fontsize=6.5, loc="lower right")
        self._save_figure(fig, "fig3_qos_compliance")

    def plot_fig4_convergence(self, convergence_data: Dict[str, List[List[float]]]):
        fig, ax = plt.subplots(figsize=(3.6, 3.2))

        for algo, curves in convergence_data.items():
            if not curves:
                continue
            arr = np.array(curves)
            mean_c = np.mean(arr, axis=0)
            n_samples = len(arr)
            std_c = np.std(arr, axis=0, ddof=1) if n_samples > 1 else np.zeros_like(mean_c)
            critical = float(t.ppf(.975, n_samples - 1)) if n_samples > 1 else 0.0
            ci_c = critical * std_c / np.sqrt(max(1, n_samples))
            iters = np.arange(len(mean_c))

            style = ALGO_STYLES.get(algo, {"color": "black", "marker": "None", "label": algo, "ls": "-"})

            ax.plot(
                iters,
                mean_c,
                label=style["label"],
                color=style["color"],
                linestyle=style["ls"],
                lw=2.0 if "WE-KB-FPA" in algo else 1.2,
            )
            ax.fill_between(
                iters,
                np.maximum(1e-1, mean_c - ci_c),
                mean_c + ci_c,
                color=style["color"],
                alpha=0.12,
            )

        ax.set_yscale("log")
        ax.set_xlabel(r"Iteration Count")
        ax.set_ylabel(r"Best System Cost ($J$) [Log Scale]")
        ax.set_title("Search Convergence Trajectories", y=1.24, fontsize=10)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=True,
            fontsize=6.0,
            columnspacing=0.8,
            handletextpad=0.3,
            handlelength=1.5,
        )
        self._save_figure(fig, "fig4_convergence_curves")

    def plot_fig5_ablation(self, df_ablation: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

        ablation_order = ["FPA-Baseline", "KB-FPA", "WE-FPA", "WE-KB-FPA"]
        sub_df = df_ablation[df_ablation["algorithm"].isin(ablation_order)]

        summary = sub_df.groupby("algorithm")["cost_j"].agg(
            mean="mean",
            ci=self._compute_ci95
        ).reindex(ablation_order).reset_index()

        colors = ["#1565C0", "#6A1B9A", "#E65100", "#B22222"]
        bars = ax.bar(
            summary["algorithm"],
            summary["mean"],
            yerr=summary["ci"],
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            width=0.6,
        )

        ax.set_yscale("log")
        ax.set_ylim(bottom=150, top=4.0e6)
        ax.set_xlim(-0.6, 3.6)
        baseline_cost = summary.loc[summary["algorithm"] == "FPA-Baseline", "mean"].values[0]
        for idx, bar in enumerate(bars):
            m_cost = summary.loc[idx, "mean"]
            y_ci = summary.loc[idx, "ci"]
            top_err = m_cost + y_ci
            y_pos = max(m_cost * 1.6, top_err * 1.35)

            if m_cost < baseline_cost:
                reduction = ((baseline_cost - m_cost) / baseline_cost) * 100.0
                label_text = f"-{reduction:.1f}%"
            else:
                exponent = int(np.floor(np.log10(m_cost)))
                coeff = m_cost / (10 ** exponent)
                label_text = rf"${coeff:.2f}\times 10^{{{exponent}}}$" + "\n(Baseline)"

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                label_text,
                ha="center",
                va="bottom",
                color="black",
                weight="bold",
                fontsize=6.8 if "\n" in label_text else 7.5,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    edgecolor="#B0BEC5",
                    linewidth=0.6,
                    alpha=0.95,
                ),
            )

        ax.set_ylabel(r"Total System Cost ($J$) [Log Scale]")
        ax.set_title("Synergistic Ablation Study")
        plt.xticks(rotation=15, ha="right", fontsize=8)
        self._save_figure(fig, "fig5_ablation_study")

    def plot_fig6_runtime_vs_exact(self, df_runtime: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))

        algos_to_plot = ["Uniform-assignment enumeration", "WE-KB-FPA", "PSO", "GA"]

        for algo in algos_to_plot:
            df_a = df_runtime[df_runtime["algorithm"] == algo]
            if df_a.empty:
                continue

            summary = df_a.groupby("parameter_value")["runtime_ms"].agg(
                mean="mean",
                ci=self._compute_ci95
            ).reset_index()
            style = ALGO_STYLES.get(algo, {"color": "black", "marker": "o", "label": algo, "ls": "-"})

            ax.plot(
                summary["parameter_value"],
                summary["mean"],
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["ls"],
                lw=2.0 if "Exact" in algo or "WE-KB-FPA" in algo else 1.2,
            )

        ax.set_yscale("log")
        ax.set_xlabel("Number of Users ($N$)")
        ax.set_ylabel("Wall-Clock Execution Time (ms) [Log Scale]")
        ax.set_title("Runtime vs. uniform-allocation reference")
        ax.legend(frameon=True, fontsize=6.5, loc="upper left")
        self._save_figure(fig, "fig6_runtime_vs_reference")
