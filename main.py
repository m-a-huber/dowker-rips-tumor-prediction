import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from scripts.compute_persistence_images import (
    compute_persistence_images,
)
from scripts.compute_persistences import compute_persistences
from scripts.compute_svm_accuracies import (
    compute_SVM_accuracies,
    get_data,
)
from scripts.process_point_clouds import process_point_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Dowker/Dowker-Rips tumor prediction pipeline.",
    )
    parser.add_argument(
        "complex",
        choices=["dowker", "dowker_rips"],
        help="Type of simplicial complex to use.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of repeated SVM evaluations (default: 10).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Number of jobs to run in parallel in hyperparameter tuning "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help=(
            "Verbosity level; values beyond 1 only affect verbosity of "
            "hyperparameter tuning."
        ),
    )
    args = parser.parse_args()
    args.complex_name = "Dowker" if args.complex == "dowker" else "Dowker-Rips"
    return args


def main(args: argparse.Namespace) -> None:
    # Get ids and times of files used in experiment
    with h5py.File("data/sim_ids_and_times.jld2", "r") as f:
        sims = f["sims"][:]
        dereferenced_sims = [f[ref][:] for ref in sims]
    ids_and_times = np.array(dereferenced_sims)
    # Get files containing point clouds used in experiment
    point_clouds_dir = Path("data/point_clouds")
    point_cloud_files = [
        (point_clouds_dir / f"ID-{id}_time-{time}_From2ParamSweep_Data.csv")
        for id, time in ids_and_times
    ]
    for point_cloud_file in tqdm(
        point_cloud_files,
        desc="Processing point clouds",
    ):
        try:
            process_point_cloud(
                point_cloud_file,
                overwrite=args.overwrite,
            )
            if args.verbose:
                tqdm.write(
                    f"Processed point cloud data at `{point_cloud_file}`."
                )
        except FileNotFoundError:
            if args.verbose:
                tqdm.write(f"File {point_cloud_file} not found, skipping.")
    # Compute persistences from processed point clouds
    processed_point_cloud_files = list(
        Path("outfiles/point_clouds_processed").iterdir()
    )
    for processed_point_cloud_file in tqdm(
        processed_point_cloud_files,
        desc=f"Computing {args.complex_name} persistences",
    ):
        compute_persistences(
            args.complex,
            processed_point_cloud_file,
            overwrite=args.overwrite,
        )
        if args.verbose:
            tqdm.write(
                f"Computed {args.complex_name} persistence of processed point "
                f"cloud at `{processed_point_cloud_file}`."
            )
    # Compute persistence images from persistences
    persistences_files = list(
        Path(f"outfiles/{args.complex}_persistences").iterdir()
    )
    for persistences_file in tqdm(
        persistences_files,
        desc=f"Computing {args.complex_name} persistence images",
    ):
        compute_persistence_images(
            args.complex,
            persistences_file,
            overwrite=args.overwrite,
        )
        if args.verbose:
            tqdm.write(
                f"Computed {args.complex_name} persistence image of "
                f"persistences at `{persistences_file}`."
            )
    # Train and evaluate SVM classifiers
    point_clouds_processed_dir = Path("outfiles/point_clouds_processed")
    persistence_images_dir = Path(
        f"outfiles/{args.complex}_persistence_images"
    )
    random_state = 42  # seed for reproduciblity
    X, y = get_data(
        point_clouds_processed_dir=point_clouds_processed_dir,
        persistence_images_dir=persistence_images_dir,
    )
    accuracies = compute_SVM_accuracies(
        X=X,
        y=y,
        complex=args.complex,
        n_repeats=args.n_repeats,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        overwrite=args.overwrite,
        random_state=random_state,
    )
    if args.verbose:
        print(f"Accuracies are: {accuracies}")
        print(
            f"Average accuracy across {args.n_repeats} runs is: "
            f"{np.around(np.mean(accuracies), 2)}"
            f"\u00b1{np.around(np.std(accuracies), 2)}."
        )
        print(
            f"Median accuracy across {args.n_repeats} runs is: "
            f"{np.around(np.median(accuracies), 2)}."
        )


if __name__ == "__main__":
    main(parse_args())
