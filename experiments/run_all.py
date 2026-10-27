"""Run selected or complete experiment suites; outputs are created on demand."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
import matplotlib
matplotlib.use("Agg")
from experiments.run_scalability import run_scalability_benchmark
from experiments.run_channel_degradation import run_channel_degradation_benchmark
from experiments.run_workload import run_workload_benchmark
from experiments.run_ablation import run_ablation_benchmark
from experiments.run_optimality_gap import run_optimality_gap_benchmark
from experiments.run_topology_shock import run_topology_shock
from we_kb_fpa.checkpoint import RunLock


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Reduced trials and sweep sizes; validation only")
    mode.add_argument("--full", action="store_true", help="Full configured sweep (default)")
    mode.add_argument("--rigorous", action="store_true", help="30 trials, up to 500 users, 30 shock seeds and 100 slots")
    parser.add_argument("--experiment", choices=["all", "scalability", "channel", "workload", "ablation", "optimality", "shock"], default="all")
    parser.add_argument("--trials", type=positive_int, help="Override trials in static sweeps")
    parser.add_argument("--training-slots", type=positive_int, help="DQN pretraining budget per shock seed")
    parser.add_argument("--slots", type=positive_int, help="Even shock evaluation horizon >= 6")
    parser.add_argument("--training-budgets", type=positive_int, nargs="+", help="Budget checkpoints; last must equal training-slots")
    parser.add_argument("--tuning-slots", type=positive_int, help="Training per candidate on two development seeds")
    parser.add_argument("--validation-slots", type=positive_int, help="Held-out slots per checkpoint validation seed")
    parser.add_argument("--validation-interval", type=positive_int)
    parser.add_argument("--fading-values", type=float, nargs="+", default=[35.])
    parser.add_argument("--require-nominal-quality", action="store_true", help="Stop before testing any DQN checkpoint that fails held-out nominal quality")
    parser.add_argument("--workers", type=positive_int, default=1, help="Parallel workers for independent static trials; timings include contention")
    parser.add_argument("--checkpoint-every", type=positive_int, default=1000, help="Save learning state every N completed slots or 60 seconds; automatic resume is enabled")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved progress and start again")
    parser.add_argument("--seeds", type=int, nargs="+", help="Shock evaluation seeds")
    args = parser.parse_args()
    slots = args.slots or (12 if args.quick else 100 if args.rigorous else 40)
    if slots < 6 or slots % 2:
        parser.error("--slots must be even and >= 6")
    trials = args.trials or (2 if args.quick else 30 if args.rigorous else 15)
    seeds = args.seeds if args.seeds is not None else ([42, 43] if args.quick else list(range(42, 72 if args.rigorous else 62)))
    training_slots = args.training_slots or (100 if args.quick else 20000 if args.rigorous else 10000)
    tuning_slots = args.tuning_slots or (16 if args.quick else 5000 if args.rigorous else 2000)
    validation_slots = args.validation_slots or (8 if args.quick else 200 if args.rigorous else 100)
    n_values = [10, 20] if args.quick else [10, 20, 50, 100, 200, 500] if args.rigorous else [10, 20, 50, 100, 200]
    plan = {
        "scalability": lambda: run_scalability_benchmark(n_values=n_values, num_trials=trials, num_workers=args.workers, resume=not args.fresh),
        "channel": lambda: run_channel_degradation_benchmark(sinr_db_values=[-5., 10.] if args.quick else [-5., 0., 5., 10., 15., 20.], num_trials=trials, num_workers=args.workers, resume=not args.fresh),
        "workload": lambda: run_workload_benchmark(d_k_mb_values=[.5, 2.5] if args.quick else [.5, 1., 1.5, 2., 2.5, 3.5, 5.], num_trials=trials, num_workers=args.workers, resume=not args.fresh),
        "ablation": lambda: run_ablation_benchmark(num_trials=args.trials or (2 if args.quick else 50 if args.rigorous else 25), num_workers=args.workers, resume=not args.fresh),
        "optimality": lambda: run_optimality_gap_benchmark(n_values=[3, 4] if args.quick else [3, 4, 5, 6, 7, 8], num_trials=args.trials or (1 if args.quick else 10 if args.rigorous else 5), num_workers=args.workers, resume=not args.fresh),
        "shock": lambda: run_topology_shock(slots=slots, training_slots=training_slots, seeds=seeds,
                    training_budgets=args.training_budgets, tuning_slots=tuning_slots,
                    validation_slots=validation_slots, validation_interval=args.validation_interval,
                    fading_values=args.fading_values, require_nominal_quality=args.require_nominal_quality,
                    checkpoint_every=args.checkpoint_every, resume=not args.fresh),
    }
    selected = list(plan) if args.experiment == "all" else [args.experiment]
    metadata = dict(arguments=vars(args), experiments=selected,
                    shock_seeds=seeds, dqn_training_slots=training_slots, sweep_trials=trials,
                    shock_slots=slots, scalability_users=n_values,
                    tuning_slots=tuning_slots, validation_slots=validation_slots,
                    python=platform.python_version(),
                    platform=platform.platform(), started_utc=datetime.now(timezone.utc).isoformat(),
                    packages={name: version(name) for name in ("numpy", "scipy", "pandas", "matplotlib", "scikit-learn")},
                    status="running", completed=[])
    with RunLock("output/checkpoints/run.lock"):
        metadata_path = Path("output/run_metadata.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        print("Automatic resume enabled." if not args.fresh else "Starting fresh; saved progress will be replaced.", flush=True)
        start = time.perf_counter()
        try:
            for name in selected:
                print(f"Running {name} ...", flush=True)
                experiment_start = time.perf_counter()
                plan[name]()
                metadata.setdefault("experiment_seconds", {})[name] = time.perf_counter() - experiment_start
                metadata["completed"].append(name)
                metadata_path.write_text(json.dumps(metadata, indent=2))
            metadata["status"] = "complete"
        except KeyboardInterrupt:
            metadata["status"] = "interrupted"
            print("Stopped. Re-run the same command to resume from the last checkpoint.", flush=True)
            raise
        except BaseException as exc:
            metadata["status"] = "failed"
            metadata["error"] = str(exc)
            raise
        finally:
            metadata["duration_seconds"] = time.perf_counter() - start
            metadata_path.write_text(json.dumps(metadata, indent=2))
        print("Results: output/data/ and output/topology_shock/; figures: output/figures/ and output/topology_shock/")


if __name__ == "__main__":
    main()
