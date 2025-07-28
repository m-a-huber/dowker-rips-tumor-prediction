import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt  # type: ignore
from dowker_complex import DowkerComplex  # type: ignore
from dowker_rips_complex import DowkerRipsComplex  # type: ignore
from sklearn.base import clone  # type: ignore


def compute_persistences(
    complex: str,
    point_cloud_file: Path,
    overwrite: bool = False,
) -> list[list[npt.NDArray[np.float64]]]:
    """Compute the required persistences from a processed point cloud and saves
    the output as a list containing the 0- and 1-dimensional homology of each
    label combination.

    Args:
        complex (str): Which complex to construct and compute persistence of.
            Must be one of `"dowker"` and `"dowker_rips"`.
        point_cloud_file (Path): Path to .npz-file containing processed point
            cloud data.
        overwrite (bool, optional): Whether or not to overwrite existing
            output. Defaults to False.

    Returns:
        list[list[npt.NDArray[np.float]]]: The persistent homologies computed
            from the deisred simplicial complex. The format of this data is a
            list of lists of NumPy-arrays of shape `(n_generators, 2)`, with
            one entry for each of the label combinations (macrophage, vessel),
            (tumor, vessel) and (macrophage, tumor). The i-th entry of each
            entry of this list is an array containing the birth and death times
            of the homological generators in dimension i-1. In particular, the
            list starts with 0-dimensional homology and contains information
            from consecutive homological dimensions.
    """
    file_out = (
        Path(f"outfiles/{complex}_persistences") / point_cloud_file.name
    ).with_suffix(".pkl")
    if not file_out.is_file() or overwrite:
        file_out.parent.mkdir(parents=True, exist_ok=True)
        persistences = []
        npz_file = np.load(point_cloud_file, allow_pickle=True)
        cells, cells_labels, _ = [npz_file[key] for key in npz_file]
        cell_label_combinations = [
            ("M", "V"),
            ("T", "V"),
            ("M", "T"),
        ]
        if complex == "dowker":
            complex = DowkerComplex(
                swap=True,
            )
        elif complex == "dowker_rips":
            complex = DowkerRipsComplex(
                n_threads=-1,
                swap=True,
            )
        else:
            raise ValueError(
                "Got invalid value for `complex`; must be one of `'dowker'`"
                "and `'dowker_rips'`."
            )
        for vertex_label, witness_label in cell_label_combinations:
            vertices = cells[cells_labels == vertex_label]
            witnesses = cells[cells_labels == witness_label]
            persistence = clone(complex).fit_transform([vertices, witnesses])
            persistences.append(persistence)
        with open(file_out, "wb") as f_out:
            pickle.dump(persistences, f_out)
    else:
        with open(file_out, "rb") as f_in:
            persistences = pickle.load(f_in)
    return persistences
