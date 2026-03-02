import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from dowker_complex import DowkerComplex
from dowker_rips_complex import DowkerRipsComplex
from dowker_rips_complex_gudhi import DowkerRipsComplexGudhi
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from tqdm import tqdm

N_POINTS_VALUES = [2 ** (i + 1) for i in range(10)]  # [2, 4, ..., 1024]
DIM_VALUES = [2 ** (i + 1) for i in range(10)]  # [2, 4, ..., 1024]
N_POINTS_BASE = 512
DIM_BASE = 512
RATIO_VERTICES = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmarking of Dowker-Rips complex implementations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vary",
        type=str,
        required=True,
        choices=["size", "dim"],
        help="Whether to vary the size or the dimension of the point cloud",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generator",
    )
    parser.add_argument(
        "--n-datasets",
        type=int,
        default=10,
        help=(
            "Number of independently generated datasets per benchmarking "
            "setting"
        ),
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help=(
            "Number of times to repeat the timing of the estimators on each "
            "dataset"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()
    if args.n_datasets <= 0:
        raise ValueError("Number of datasets must be positive")
    if args.n_repeats <= 0:
        raise ValueError("Number of repeats must be positive")
    return args


def generate_datasets(
    n_points: int,
    dim: int,
    ratio_vertices: float,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return train_test_split(
        rng.standard_normal(size=(n_points, dim)),
        train_size=ratio_vertices,
        random_state=rng.integers(low=0, high=2**32),
    )


def time_fit_transform(
    estimator: DowkerComplex | DowkerRipsComplex | DowkerRipsComplexGudhi,
    dataset: list[np.ndarray],
    n_repeats: int,
) -> np.ndarray:
    # Exclude warm-up run from timing.
    warmup_dataset = [arr.copy() for arr in dataset]
    clone(estimator).fit_transform(warmup_dataset)
    elapsed_times = []
    for _ in range(n_repeats):
        dataset_copied = [arr.copy() for arr in dataset]
        start = time.perf_counter()
        clone(estimator).fit_transform(dataset_copied)
        elapsed_times.append(time.perf_counter() - start)
    return np.asarray(elapsed_times)


def main(
    vary: str,
    n_points_values: list[int],
    dim_values: list[int],
    n_points_base: int,
    dim_base: int,
    ratio_vertices: float,
    seed: int,
    n_datasets: int,
    n_repeats: int,
    verbose: bool,
    overwrite: bool,
) -> None:
    outfile = Path(
        f"benchmarking_results/benchmarking_results_vary_{vary}_{n_datasets}_datasets_{n_repeats}_repeats_seed_{seed}.csv"
    )
    if outfile.exists() and not overwrite:
        df = pl.read_csv(outfile)
        if verbose:
            tqdm.write(
                f"Found benchmarking results at {outfile}; not overwriting."
            )
    else:
        configs: list[
            tuple[
                str, DowkerComplex | DowkerRipsComplex | DowkerRipsComplexGudhi
            ]
        ] = [
            ("DowkerComplex", DowkerComplex(swap=False)),
            (
                "DowkerRipsComplex(use_numpy=False)",
                DowkerRipsComplex(swap=False, use_numpy=False),
            ),
            (
                "DowkerRipsComplex(use_numpy=True)",
                DowkerRipsComplex(swap=False, use_numpy=True),
            ),
            ("DowkerRipsComplexGudhi", DowkerRipsComplexGudhi(swap=False)),
        ]
        varied_values = n_points_values if vary == "size" else dim_values
        if verbose:
            tqdm.write(
                f"Running {len(configs) * len(varied_values)} benchmarks "
                f"({len(configs)} configs \u00d7 "
                f"{len(varied_values)} {vary} values)"
            )
        df = pl.DataFrame(
            schema={
                "config": pl.Utf8,
                "n_points": pl.Int64,
                "dim": pl.Int64,
                "n_vertices": pl.Int64,
                "n_witnesses": pl.Int64,
                "n_datasets": pl.Int64,
                "time_mean": pl.Float64,
                "time_std": pl.Float64,
                "time_median": pl.Float64,
                "time_iqr": pl.Float64,
                "dataset_mean_std": pl.Float64,
            }
        )
        if vary == "size":
            benchmark_inputs = [
                (n_points, dim_base) for n_points in n_points_values
            ]
            desc = "Running Dowker-Rips benchmarking (varying size)"
        else:
            benchmark_inputs = [(n_points_base, dim) for dim in dim_values]
            desc = "Running Dowker-Rips benchmarking (varying dimension)"
        order_rng = np.random.default_rng(seed)
        for n_points, dim in tqdm(benchmark_inputs, desc=desc):
            if verbose:
                tqdm.write(
                    f"Running benchmarking for n_points={n_points}, "
                    f"dim={dim}..."
                )
            datasets = [
                generate_datasets(
                    n_points=n_points,
                    dim=dim,
                    ratio_vertices=ratio_vertices,
                    seed=seed + i,
                )
                for i in range(n_datasets)
            ]
            elapsed_times_by_config: dict[str, list[np.ndarray]] = {
                label: [] for label, _ in configs
            }
            dataset_means_by_config: dict[str, list[float]] = {
                label: [] for label, _ in configs
            }
            for dataset in datasets:
                ordered_configs = [
                    configs[i] for i in order_rng.permutation(len(configs))
                ]
                for label, estimator in ordered_configs:
                    elapsed_times = time_fit_transform(
                        estimator=estimator,
                        dataset=dataset,
                        n_repeats=n_repeats,
                    )
                    elapsed_times_by_config[label].append(elapsed_times)
                    dataset_means_by_config[label].append(
                        float(np.mean(elapsed_times))
                    )
            for label, _ in configs:
                elapsed_times_all = elapsed_times_by_config[label]
                dataset_means = dataset_means_by_config[label]
                elapsed_times_flat = np.concatenate(elapsed_times_all)
                row = pl.DataFrame(
                    {
                        "config": [label],
                        "n_points": [n_points],
                        "dim": [dim],
                        "n_vertices": [datasets[0][0].shape[0]],
                        "n_witnesses": [datasets[0][1].shape[0]],
                        "n_datasets": [n_datasets],
                        "time_mean": [float(np.mean(elapsed_times_flat))],
                        "time_std": [float(np.std(elapsed_times_flat))],
                        "time_median": [float(np.median(elapsed_times_flat))],
                        "time_iqr": [
                            float(
                                np.percentile(elapsed_times_flat, 75)
                                - np.percentile(elapsed_times_flat, 25)
                            )
                        ],
                        "dataset_mean_std": [float(np.std(dataset_means))],
                    }
                )
                df = df.vstack(row)
        # Save results to CSV
        outfile.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(outfile)
        if verbose:
            tqdm.write(f"Saved benchmarking results to {outfile}.")
    if verbose:
        tqdm.write("Benchmarking results:")
        tqdm.write(str(df))


if __name__ == "__main__":
    args = parse_args()
    main(
        vary=args.vary,
        n_points_values=N_POINTS_VALUES,
        dim_values=DIM_VALUES,
        n_points_base=N_POINTS_BASE,
        dim_base=DIM_BASE,
        ratio_vertices=RATIO_VERTICES,
        seed=args.seed,
        n_datasets=args.n_datasets,
        n_repeats=args.n_repeats,
        verbose=args.verbose,
        overwrite=args.overwrite,
    )
