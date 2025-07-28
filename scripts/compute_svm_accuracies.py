from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt  # type: ignore
from imblearn.over_sampling import SMOTE  # type: ignore
from imblearn.pipeline import Pipeline  # type: ignore
from scipy.stats import loguniform  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.model_selection import (  # type: ignore
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.svm import SVC  # type: ignore
from tqdm import tqdm  # type: ignore
from typing_extensions import Self


# Custom transformer to mimic UnitRangeTransform from Julia
class _UnitRangeTransform(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        verbose: bool,
    ):
        self.verbose = verbose

    def fit(
        self,
        X: npt.NDArray,
        y: Optional[None] = None,
    ) -> Self:
        if self.verbose:
            print("Fitting _UnitRangeTransform...")
        self.min_ = X.min(axis=0)
        self.range_ = X.max(axis=0) - self.min_
        if self.verbose:
            print("Done fitting _UnitRangeTransform.")
        return self

    def transform(
        self,
        X: npt.NDArray,
        y: Optional[None] = None,
    ):
        if self.verbose:
            print("Transforming data using _UnitRangeTransform...")
        X_scaled = (X - self.min_) / self.range_
        if self.verbose:
            print("Done transforming data using _UnitRangeTransform...")
        return X_scaled


def get_data(
    point_clouds_processed_dir: Path,
    persistence_images_dir: Path,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Fetches data used for training and evaluating of SVMs.

    Args:
        point_clouds_processed_dir (Path): Directory containing the processed
            point clouds.
        persistence_images_dir (Path): Directory containing the concatenated
            persistence images.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: Tuple of two arrays, the first one
            containing the concatenated persistence images corresponding to a
            point cloud, and the second one the label encoding M1/M2-dominance.
    """
    X_list: list[npt.NDArray] = []
    y_list: list[int] = []
    for persistence_images_path in sorted(persistence_images_dir.iterdir()):
        persistence_images = np.load(persistence_images_path)
        persistence_images_concat = np.concatenate(persistence_images)
        X_list.append(persistence_images_concat)
        point_cloud_path = (
            point_clouds_processed_dir / persistence_images_path.name
        ).with_suffix(".npz")
        npz_file = np.load(point_cloud_path, allow_pickle=True)
        _, _, point_cloud_label = [npz_file[key] for key in npz_file]
        y_list.append(int(point_cloud_label))
    X, y = np.array(X_list), np.array(y_list)
    return X, y


def _train_svm(
    X: npt.NDArray,
    y: npt.NDArray,
    n_jobs: int,
    verbose: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=rng.integers(0, 1 << 32)
    )
    param_dist = {
        "svc__C": loguniform(1e-6, 1000.0),
        "svc__gamma": loguniform(1e-6, 1000.0),
    }
    sm = SMOTE(random_state=rng.integers(0, 1 << 32))
    clf = SVC()
    svm_pipeline = Pipeline([("smote", sm), ("svc", clf)])
    random_search = RandomizedSearchCV(
        svm_pipeline,
        param_dist,
        n_iter=500,
        n_jobs=n_jobs,
        random_state=rng.integers(0, 1 << 32),
        refit=True,
        verbose=verbose,
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    C = best_params.get("svc__C", "Error")
    gamma = best_params.get("svc__gamma", "Error")
    return C, gamma


def _repeat_svm(
    X: npt.NDArray,
    y: npt.NDArray,
    C: float,
    gamma: float,
    n_repeats: int,
    rng: np.random.Generator,
) -> npt.NDArray:
    accuracies = []
    for _ in tqdm(range(n_repeats), desc="Fitting SVMs"):
        shuffled_ixs = rng.permutation(len(X))
        X_shuffled = X[shuffled_ixs]
        y_shuffled = y[shuffled_ixs]
        X_train, X_test, y_train, y_test = train_test_split(
            X_shuffled,
            y_shuffled,
            test_size=0.3,
            random_state=rng.integers(0, 1 << 32),
        )
        sm = SMOTE(random_state=rng.integers(0, 1 << 32))
        clf = SVC(C=C, gamma=gamma)
        svm_pipeline = Pipeline([("smote", sm), ("svc", clf)])
        svm_pipeline.fit(X_train, y_train)
        y_pred = svm_pipeline.predict(X_test)
        acc = np.mean(y_pred == y_test) * 100
        accuracies.append(acc)
    return np.array(accuracies)


def compute_SVM_accuracies(
    X: npt.NDArray,
    y: npt.NDArray,
    complex: str,
    n_repeats: int,
    n_jobs: int,
    verbose: int,
    overwrite: bool,
    random_state: Optional[int] = None
) -> npt.NDArray:
    """Optimizes hyperparameters of a SVM on a train portion of `X`, and then
    trains and evaluates an SVM with the parameters found on a train and test
    portion of `X` respectively. This latter portion is performed `n_repeats`
    times using different splits.

    Args:
        X (npt.NDArray): Array containing concatenated persistence images.
        y (npt.NDArray): Array containing label encoding M1/M2-dominance.
        complex (str): Which complex to use. Must be one of `'dowker'` and
            `'dowker_rips'`.
        n_repeats (int): Number of times to repeat training and evaluating of
            SVM with the optimal hyperparameters found.
        n_jobs (int): Number of jobs to run in parllel during hyperparameter
            tuning with `RandomizedSearchCV`.
        verbose (int): Level of verbosity. The higher the value, the more
            information is displayed.
        overwrite (bool): Whether or not to overwrite existing files produced
            in the process.
        random_state (Optional[int], optional): A seed allowing for
            reproducible results. Defaults to None.

    Returns:
        npt.NDArray: NumPy-array of shape `(n_repeats,)` containing the
            accuracies of each SVM trained.
    """
    file_out = Path(f"outfiles/accuracies_{complex}_{n_repeats}_runs.npy")
    if not file_out.is_file() or overwrite:
        rng = np.random.default_rng(random_state)
        scaler = _UnitRangeTransform(verbose=bool(verbose))
        X_scaled = scaler.fit_transform(X)
        C, gamma = _train_svm(
            X=X_scaled, y=y, n_jobs=n_jobs, verbose=verbose, rng=rng
        )
        accuracies = _repeat_svm(
            X=X_scaled,
            y=y,
            C=C,
            gamma=gamma,
            n_repeats=n_repeats,
            rng=rng
        )
        np.save(file_out, accuracies)
    else:
        accuracies = np.load(file_out)
    return accuracies
