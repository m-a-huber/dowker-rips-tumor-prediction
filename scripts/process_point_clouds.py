from pathlib import Path

import numpy as np
import numpy.typing as npt  # type: ignore
import pandas as pd  # type: ignore


def process_point_cloud(
    point_cloud_file: Path,
    overwrite: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.str_], int]:
    """Processes file containing point cloud and produces and saves the output
    in two NumPy-arrays that contain the xy-coordinates and the labels of the
    cells, respectively, as well as an `int` representing the M1/M2-dominance
    of the point cloud.

    Args:
        point_cloud_file (Path): Path to CSV-file containing point cloud data.
        overwrite (bool, optional): Whether or not to overwrite existing
            output. Defaults to False.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.str_], int]: Tuple
            containing
            - a NumPy-array of shape (n_cells, 2) containing the xy-coordinates
            of the cells;
            - a NumPy-array of shape (n_cells,) containing the labels of the
            cells, where each label is one of `"M"`, `"T"`, `"V"` and `"N"`;
            - an integer representing the M1/M2-dominance of the point-cloud,
            where `0` and `1` correspond to M1- and M2-dominance, respectively.
    """
    file_out = (
        Path("outfiles/point_clouds_processed") / point_cloud_file.name
    ).with_suffix(".npz")
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
            cells=cells,
            cells_labels=cells_labels,
            point_cloud_label=point_cloud_label,
        )
    else:
        npz_file = np.load(file_out, allow_pickle=True)
        cells = npz_file["cells"]
        cells_labels = npz_file["cells_labels"]
        point_cloud_label = npz_file["point_cloud_label"]
    return cells, cells_labels, point_cloud_label
