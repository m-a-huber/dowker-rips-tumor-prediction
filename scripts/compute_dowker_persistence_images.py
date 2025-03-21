import pickle
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt  # type: ignore
from gudhi.representations import PersistenceImage  # type: ignore
from tqdm import tqdm  # type: ignore


def drop_infty(dim):
    return dim[
        ~np.isinf(dim).any(axis=1)
    ]


def weight(pt):  # Default weight function used in `PersistenceDiagrams.jl`
    return min(1, pt[1])


def compute_persistence_images(
        persistences_file: Path,
        overwrite: bool = False,
    ) -> npt.NDArray[np.float64]:
    """Compute and concatenate the required persistence landscapes of the
    Dowker persistences and saves the output as a NumPy-array.

    Args:
        persistences_file (Path): Path to .pkl-file containing Dowker
            persistences.
        overwrite (bool, optional): Whether or not to overwrite existing
            output. Defaults to False.

    Returns:
        npt.NDArray[np.float64]: NumPy-array of shape (6, 400) that is obtained
            as the concatenation of the six flattened persistence images.
    """
    file_out = (
        Path("outfiles/dowker_persistence_images")
        / persistences_file.name
    ).with_suffix(".npy")
    if not file_out.is_file() or overwrite:
        file_out.parent.mkdir(parents=True, exist_ok=True)
        with open(persistences_file, "rb") as f_in:
            persistences = pickle.load(f_in)
        persistences_flattened = [
            drop_infty(dim)
            for persistence in persistences
            for dim in persistence
        ]
        persistence_images = PersistenceImage(
            bandwidth=1,
            resolution=[20, 20],
            weight=weight,
        ).fit_transform(persistences_flattened)
        np.save(file_out, persistence_images)
    else:
        persistence_images = np.load(file_out)
    return persistence_images


if __name__ == "__main__":
    persistences_files = list(
        Path("outfiles/dowker_persistences").iterdir()
    )
    overwrite, verbose = sys.argv[1] == "True", sys.argv[2] == "True"
    for persistences_file in tqdm(
        persistences_files,
        desc="Computing Dowker persistence images",
    ):
        compute_persistence_images(
            persistences_file,
            overwrite=overwrite,
        )
        if verbose:
            tqdm.write(
                "Computed Dowker persistence image of persistences "
                f"at `{persistences_file}`."
            )
