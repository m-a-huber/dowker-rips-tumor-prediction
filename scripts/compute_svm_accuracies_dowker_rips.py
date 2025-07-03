import pickle
import sys
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


# Custom transformer to mimic UnitRangeTransform
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
    persistence_images_dir: Path,
) -> tuple[npt.NDArray, npt.NDArray]:
    X_list: list[npt.NDArray] = []
    y_list: list[int] = []
    for persistence_images_path in persistence_images_dir.iterdir():
        persistence_images = np.load(persistence_images_path)
        persistence_images_concat = np.concatenate(persistence_images)
        X_list.append(persistence_images_concat)
        point_cloud_path = (
            Path("outfiles/point_clouds_processed")
            / persistence_images_path.name
        ).with_suffix(".npz")
        npz_file = np.load(point_cloud_path, allow_pickle=True)
        cells, cells_labels, point_cloud_label = [
            npz_file[key]
            for key in npz_file
        ]
        y_list.append(int(point_cloud_label))
    X, y = np.array(X_list), np.array(y_list)
    return X, y


def _train_svm(
        X: npt.NDArray,
        y: npt.NDArray,
        verbose: int,
    ) -> tuple[float, float, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    param_dist = {
        "svc__C": loguniform(1e-6, 1000.0),
        "svc__gamma": loguniform(1e-6, 1000.0),
    }
    sm = SMOTE(random_state=42)
    clf = SVC()
    svm_pipeline = Pipeline([("smote", sm), ("svc", clf)])
    random_search = RandomizedSearchCV(
        svm_pipeline,
        param_dist,
        n_iter=500,
        random_state=41,
        refit=True,
        verbose=verbose,
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    C = best_params.get("svc__C", "Error")
    gamma = best_params.get("svc__gamma", "Error")
    score = random_search.score(X_test, y_test)
    return score, C, gamma


def _repeat_svm(
    X: npt.NDArray,
    y: npt.NDArray,
    C: float,
    gamma: float,
    n_repeats: int = 10
) -> npt.NDArray:
    accuracies = []
    for _ in tqdm(
        range(n_repeats),
        desc="Fitting SVMs"
    ):
        shuffled_ixs = np.random.permutation(len(X))
        X = X[shuffled_ixs]
        y = y[shuffled_ixs]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        sm = SMOTE(random_state=41)
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
        verbose: int,
        overwrite: bool = False,
    ) -> npt.NDArray:
    file_out = Path("outfiles/accuracies_dowker_rips.pkl")
    if not file_out.is_file() or overwrite:
        scaler = _UnitRangeTransform(verbose=bool(verbose))
        X_scaled = scaler.fit_transform(X)
        score, C, gamma = _train_svm(X_scaled, y, verbose)
        accuracies = _repeat_svm(
            X_scaled,
            y,
            C,
            gamma,
            10
        )
        with open(file_out, "wb") as f_out:
            pickle.dump(accuracies, f_out)
    else:
        with open(file_out, "rb") as f_in:
            accuracies = pickle.load(f_in)
    return accuracies


if __name__ == "__main__":
    overwrite, verbose = sys.argv[1] == "True", int(sys.argv[2])
    persistence_images_dir = Path("outfiles/dowker_rips_persistence_images")
    X, y = get_data(persistence_images_dir)
    accuracies = compute_SVM_accuracies(
        X,
        y,
        verbose=verbose,
        overwrite=overwrite,
    )
    if verbose:
        print(f"Accuracies are: {accuracies}")
