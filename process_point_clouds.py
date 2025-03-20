import sys
from pathlib import Path

import h5py  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore

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
files = [
    (
        point_clouds_path
        / f"ID-{id}_time-{time}_From2ParamSweep_Data.csv"
    )
    for id, time in ids_and_times
]


def process_point_cloud(
        point_cloud_file: Path,
        overwrite: bool = False,
    ) -> None:
    """Processes file containing point cloud and produces and saves two
    NumPy-arrays that contain the xy-coordinates and the labels of the cells,
    respectively, as well as an `int` representing the M1/M2-dominance of the
    point cloud. The labels of the cells are one of `"M"`, `"T"`, `"V"` and
    `"N"`. For the label of the point cloud, `0` and `1` correspond to M1- and
    M2-dominance, respectively.
    """
    file_out = Path(
        str(point_cloud_file.with_suffix(".npz")).replace(
            "point_clouds", "point_clouds_processed"
        )
    )
    if not file_out.is_file() or overwrite:
        file_out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(point_cloud_file)
        cells = df[["points_x", "points_y"]].to_numpy()
        cells_labels = df["celltypes"].str[0].to_numpy()
        df_macrophages = df[df["celltypes"] == "Macrophage"]
        n = len(df_macrophages)
        M1_count = (
            (df_macrophages["phenotypes"] >= 0.0)
            & (df_macrophages["phenotypes"] <= 0.5)
        ).sum()
        point_cloud_label = int(n > 0 and M1_count / n < 0.5)
        np.savez(
            file_out,
            *[
                cells,
                cells_labels,
                point_cloud_label,
            ],
        )
    else:
        npz_file = np.load(file_out, allow_pickle=True)
        cells, cells_labels, point_cloud_label = [
            npz_file[key]
            for key in npz_file
        ]
    return


if __name__ == "__main__":
    overwrite, verbose = sys.argv[1] == "True", sys.argv[2] == "True"
    for file in tqdm(files, desc="Processing data"):
        try:
            process_point_cloud(
                file,
                overwrite=overwrite,
            )
            if verbose:
                tqdm.write(
                    f"Processed point cloud data at `{file}`."
                )
        except FileNotFoundError:
            if verbose:
                tqdm.write(f"File {file} not found, skipping.")
