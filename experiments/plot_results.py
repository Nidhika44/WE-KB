"""Regenerate paper figures from exported CSV/JSON data without simulations."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from we_kb_fpa.publication_plots import PublicationPlotter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    plotter = PublicationPlotter(str(args.output_dir / "figures"))
    data = args.output_dir / "data"
    sources = [
        ("scalability_benchmark.csv", plotter.plot_fig1_scalability),
        ("channel_resilience_benchmark.csv", plotter.plot_fig2_sinr_resilience),
        ("workload_qos_benchmark.csv", plotter.plot_fig3_qos_compliance),
        ("formal_ablation_benchmark.csv", plotter.plot_fig5_ablation),
        ("optimality_gap_benchmark.csv", plotter.plot_fig6_runtime_vs_exact),
    ]
    generated = 0
    for filename, render in sources:
        path = data / filename
        if path.exists():
            render(pd.read_csv(path))
            generated += 1
    convergence = data / "ablation_convergence_convergence.json"
    if convergence.exists():
        plotter.plot_fig4_convergence(json.loads(convergence.read_text()))
        generated += 1
    shock_dir = args.output_dir / "topology_shock"
    if (shock_dir / "slots.csv").exists():
        from experiments.run_topology_shock import plot_adaptation
        config = json.loads((shock_dir / "config.json").read_text())
        plot_adaptation(pd.read_csv(shock_dir / "slots.csv"), config["slots"],
                        config["fading_db"], shock_dir)
        if (shock_dir / "validation.csv").exists():
            from experiments.run_topology_shock import plot_learning
            plot_learning(pd.read_csv(shock_dir / "validation.csv"),
                          pd.read_csv(shock_dir / "slots.csv"), shock_dir)
        generated += 1
    if not generated:
        parser.error(f"No result data found under {args.output_dir}; run experiments first")
    print(f"Regenerated {generated} figure groups under {args.output_dir}")


if __name__ == "__main__":
    main()
