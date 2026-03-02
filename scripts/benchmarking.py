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
DIM = 100
RATIO_VERTICES = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmarking of Dowker-Rips complex implementations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generator.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def generate_data(
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


def time_fit_transform(estimator, X: list[np.ndarray]) -> float:
    start = time.perf_counter()
    clone(estimator).fit_transform(X)
    return time.perf_counter() - start


def main(
    n_points_values: list[int],
    dim: int,
    ratio_vertices: float,
    seed: int,
    verbose: bool,
    overwrite: bool,
) -> None:
    outfile = Path("outfiles/benchmark_results.csv")
    if outfile.exists() and not overwrite:
        df = pl.read_csv(outfile)
        if verbose:
            tqdm.write(
                f"Found benchmarking results at {outfile}; not overwriting."
            )
    else:
        configs: list[tuple[str, object]] = [
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
        if verbose:
            tqdm.write(
                f"Running {len(configs) * len(n_points_values)} benchmarks "
                f"({len(configs)} configs \u00d7 {len(n_points_values)} sizes)"
            )
        df = pl.DataFrame(
            schema={
                "config": pl.Utf8,
                "n_points": pl.Int64,
                "n_vertices": pl.Int64,
                "n_witnesses": pl.Int64,
                "time_s": pl.Float64,
            }
        )
        for n_points in tqdm(
            n_points_values, desc="Running Dowker-Rips benchmarks"
        ):
            if verbose:
                tqdm.write(f"Running benchmarks for n_points={n_points}...")
            X = generate_data(
                n_points=n_points,
                dim=dim,
                ratio_vertices=ratio_vertices,
                seed=seed,
            )
            for label, estimator in configs:
                elapsed = time_fit_transform(estimator, X)
                row_df = pl.DataFrame(
                    {
                        "config": [label],
                        "n_points": [n_points],
                        "n_vertices": [X[0].shape[0]],
                        "n_witnesses": [X[1].shape[0]],
                        "time_s": [elapsed],
                    }
                )
                df = df.vstack(row_df)
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
        n_points_values=N_POINTS_VALUES,
        dim=DIM,
        ratio_vertices=RATIO_VERTICES,
        seed=args.seed,
        verbose=args.verbose,
        overwrite=args.overwrite,
    )
