import sys
from pathlib import Path

import h5py  # type: ignore
import numpy as np
from tqdm import tqdm  # type: ignore

from scripts.compute_persistence_images import (
    compute_persistence_images,
)
from scripts.compute_persistences import compute_persistences
from scripts.compute_svm_accuracies import (
    compute_SVM_accuracies,
    get_data,
)
from scripts.process_point_clouds import process_point_cloud

if __name__ == "__main__":
    complex, overwrite, verbose = (
        sys.argv[1],
        sys.argv[2] == "True",
        int(sys.argv[3]),
    )
    if complex == "dowker":
        complex_name = "Dowker"
    elif complex == "dowker_rips":
        complex_name = "Dowker-Rips"
    else:
        raise ValueError(
            "Got invalid value for `complex`; must be one of `'dowker'`"
            "and `'dowker_rips'`."
        )

    # Get ids and times of files used in experiment
    with h5py.File("data/sim_ids_and_times.jld2", "r") as f:
        sims = f["sims"][:]
        dereferenced_sims = [
            f[ref][:]
            for ref in sims
        ]
    ids_and_times = np.array(dereferenced_sims)
    # Get files containing point clouds used in experiment
    point_clouds_path = Path("data/point_clouds")
    point_cloud_files = [
        (
            point_clouds_path
            / f"ID-{id}_time-{time}_From2ParamSweep_Data.csv"
        )
        for id, time in ids_and_times
    ]
    for point_cloud_file in tqdm(
        point_cloud_files,
        desc="Processing point clouds",
    ):
        try:
            process_point_cloud(
                point_cloud_file,
                overwrite=overwrite,
            )
            if verbose:
                tqdm.write(
                    f"Processed point cloud data at `{point_cloud_file}`."
                )
        except FileNotFoundError:
            if verbose:
                tqdm.write(f"File {point_cloud_file} not found, skipping.")

    processed_point_cloud_files = list(
        Path("outfiles/point_clouds_processed").iterdir()
    )
    for processed_point_cloud_file in tqdm(
        processed_point_cloud_files,
        desc=f"Computing {complex_name} persistences",
    ):
        compute_persistences(
            complex,
            processed_point_cloud_file,
            overwrite=overwrite,
        )
        if verbose:
            tqdm.write(
                f"Computed {complex_name} persistence of processed point "
                f"cloud at `{processed_point_cloud_file}`."
            )

    persistences_files = list(
        Path(f"outfiles/{complex}_persistences").iterdir()
    )
    for persistences_file in tqdm(
        persistences_files,
        desc=f"Computing {complex_name} persistence images",
    ):
        compute_persistence_images(
            complex,
            persistences_file,
            overwrite=overwrite,
        )
        if verbose:
            tqdm.write(
                f"Computed {complex_name} persistence image of persistences "
                f"at `{persistences_file}`."
            )

    persistence_images_dir = Path(f"outfiles/{complex}_persistence_images")
    X, y = get_data(persistence_images_dir)
    accuracies = compute_SVM_accuracies(
        X,
        y,
        complex,
        verbose=verbose,
        overwrite=overwrite,
    )
    if verbose:
        print(f"Accuracies are: {accuracies}")
