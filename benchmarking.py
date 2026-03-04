import argparse
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from dowker_complex import DowkerComplex
from dowker_rips_complex import DowkerRipsComplex
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange

from scripts import cs_wong
from scripts.dowker_rips_complex_gudhi import DowkerRipsComplexGudhi

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
        "--max-exponent",
        type=int,
        default=11,
        help=(
            "Maximum exponent for the range of point cloud size and dimension "
            "values to benchmark"
        ),
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
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generator",
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
    if args.max_exponent <= 0:
        raise ValueError("Maximum exponent must be positive")
    if args.n_datasets <= 0:
        raise ValueError("Number of datasets must be positive")
    if args.n_repeats <= 0:
        raise ValueError("Number of repeats must be positive")
    return args


def generate_dataset(
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
    for _ in trange(
        n_repeats, desc="Iterating over repeats", leave=False, position=3
    ):
        dataset_copied = [arr.copy() for arr in dataset]
        start = time.perf_counter()
        clone(estimator).fit_transform(dataset_copied)
        elapsed_times.append(time.perf_counter() - start)
    return np.asarray(elapsed_times)


def make_plot(
    df: pl.DataFrame,
    vary: str,
) -> go.Figure:
    x_col = "n_points" if vary == "size" else "dim"
    x_title = "Point cloud size" if vary == "size" else "Point cloud dimension"
    x_values = sorted(df[x_col].unique().to_list())
    configs = (  # order configs by average runtime
        df.group_by("config")
        .agg(pl.col("time_mean").mean().alias("avg_time_mean"))
        .sort("avg_time_mean", descending=False)
    )["config"].to_list()
    rgbs_wong = [f"rgb{rgb}" for rgb in cs_wong.rgbs]

    def rgbas_wong(alpha: float) -> list[str]:
        return [f"rgba{(*rgb, alpha)}" for rgb in cs_wong.rgbs]

    shapes = ["circle", "square", "diamond", "x"]
    fig = go.Figure()
    for i, config in enumerate(configs):
        df_config = df.filter(pl.col("config") == config).sort(x_col)
        y_values = df_config["time_mean"].to_list()
        dataset_mean_stds = df_config["dataset_mean_std"].to_list()
        y_upper = [
            y_value + dataset_mean_std
            for y_value, dataset_mean_std in zip(y_values, dataset_mean_stds)
        ]
        y_lower = [
            y_value - dataset_mean_std
            # max(1e-12, y_value - dataset_mean_std)
            for y_value, dataset_mean_std in zip(y_values, dataset_mean_stds)
        ]
        x_data = df_config[x_col].to_list()
        # Draw std-dev band
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_lower,
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_upper,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=rgbas_wong(alpha=0.2)[i],
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Draw mean line
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_values,
                mode="lines+markers",
                line={"color": rgbs_wong[i]},
                marker={
                    "color": rgbs_wong[i],
                    "symbol": shapes[i],
                },
                name=config,
            )
        )
    fig.update_layout(
        title="Log-log plot of mean runtimes with std. dev. across datasets",
        xaxis={
            "title": f"{x_title}",
            "type": "log",
            "tickmode": "array",
            "tickvals": x_values,
            "ticktext": [str(value) for value in x_values],
        },
        yaxis={
            "title": "Runtime (seconds)",
            "type": "log",
            "tickmode": "linear",
            "tick0": 0,
            "dtick": 1,
            "exponentformat": "power",
            "showexponent": "all",
        },
        legend_title="Algorithm configuration",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": -0.2,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"b": 140},
        template="plotly_white",
    )
    return fig


def main(
    vary: str,
    max_exponent: int,
    n_datasets: int,
    n_repeats: int,
    n_points_base: int,
    dim_base: int,
    ratio_vertices: float,
    seed: int,
    verbose: bool,
    overwrite: bool,
) -> None:
    outfile = Path(
        f"benchmarking_results/benchmarking_results"
        f"_max_exponent_{max_exponent}"
        f"_vary_{vary}"
        f"_n_datasets_{n_datasets}"
        f"_n_repeats_{n_repeats}"
        f"_seed_{seed}.csv"
    )
    if outfile.exists() and not overwrite:
        df = pl.read_csv(outfile)
        if verbose:
            tqdm.write(
                f"Found benchmarking results at {outfile}; not overwriting."
            )
    else:
        n_points_values = [2 ** (i + 1) for i in range(max_exponent)]
        dim_values = [2 ** (i + 1) for i in range(max_exponent)]
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
        # Iterate over benchmark inputs (n_points, dim)
        for n_points, dim in tqdm(benchmark_inputs, desc=desc, position=0):
            if verbose:
                tqdm.write(
                    f"Running benchmarking for n_points={n_points}, "
                    f"dim={dim}..."
                )
            datasets = [
                generate_dataset(
                    n_points=n_points,
                    dim=dim,
                    ratio_vertices=ratio_vertices,
                    seed=seed + i,
                )
                for i in range(n_datasets)
            ]
            # Initialize dicts for elapsed times and dataset means by config
            elapsed_times_by_config: dict[str, list[np.ndarray]] = {
                label: [] for label, _ in configs
            }
            dataset_means_by_config: dict[str, list[float]] = {
                label: [] for label, _ in configs
            }
            # Iterate over datasets
            for dataset in tqdm(
                datasets,
                desc="Iterating over datasets",
                leave=False,
                position=1,
            ):
                # Shuffle configs
                configs_shuffled = [
                    configs[i] for i in order_rng.permutation(len(configs))
                ]
                # Iterate over configs
                for label, estimator in tqdm(
                    configs_shuffled,
                    desc="Iterating over configs",
                    leave=False,
                    position=2,
                ):
                    # Time fit_transform
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
    plotfile_pdf = outfile.with_suffix(".pdf")
    plotfile_svg = outfile.with_suffix(".svg")
    if plotfile_pdf.exists() and plotfile_svg.exists() and not overwrite:
        if verbose:
            tqdm.write(
                f"Found PDF- and SVG-plots at {plotfile_pdf} and "
                f"{plotfile_svg}, respectively; not overwriting."
            )
    else:
        fig = make_plot(
            df,
            vary=vary,
        )
        fig.write_image(plotfile_pdf)
        fig.write_image(plotfile_svg)
        if verbose:
            tqdm.write(
                f"Saved PDF- and SVG-plots to {plotfile_pdf} and "
                f"{plotfile_svg}, respectively."
            )


if __name__ == "__main__":
    args = parse_args()
    main(
        vary=args.vary,
        max_exponent=args.max_exponent,
        n_datasets=args.n_datasets,
        n_repeats=args.n_repeats,
        n_points_base=N_POINTS_BASE,
        dim_base=DIM_BASE,
        ratio_vertices=RATIO_VERTICES,
        seed=args.seed,
        verbose=args.verbose,
        overwrite=args.overwrite,
    )
