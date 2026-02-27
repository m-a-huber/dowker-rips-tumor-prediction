import numpy as np
import pytest
from dowker_rips_complex_gudhi import DowkerRipsComplexGudhi
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)


@pytest.fixture
def random_data():
    n, dim = 100, 4
    ratio_vertices = 0.9
    X, y = (
        list(
            train_test_split(
                rng.standard_normal(size=(n, dim)), train_size=ratio_vertices
            )
        ),
        None,
    )
    return X, y


@pytest.fixture
def quadrilateral():
    vertices = np.array([[0, 0], [2, 0], [4, 2], [0, 4]])
    witnesses = np.array([[2, 3], [0, 2], [1, 0], [3, 1]])
    X, y = [vertices, witnesses], None
    return X, y


@pytest.fixture
def octagon():
    t = 1 / np.sqrt(2)
    vertices = np.array([[1, 0], [t, t], [0, 1], [-t, t]])
    witnesses = np.array([[-1, 0], [-t, -t], [0, -1], [t, -t]])
    X, y = [vertices, witnesses], None
    return X, y


def test_dowker_rips_complex_gudhi(random_data):
    """
    Check whether `DowkerRipsComplexGudhi` runs at all for `max_dimension` up
    to and including `1`.
    """
    X, y = random_data
    for max_dimension in [0, 1]:
        drcg = DowkerRipsComplexGudhi(max_dimension=max_dimension)
        drcg.fit_transform(X, y)
        assert hasattr(drcg, "persistence_")
        assert hasattr(drcg, "complex_")
        assert drcg.complex_.dimension() == max_dimension + 1


def test_dowker_rips_complex_gudhi_cosine(random_data):
    """
    Check whether `DowkerRipsComplexGudhi` runs on random data with non-default
    metric.
    """
    X, y = random_data
    drcg = DowkerRipsComplexGudhi(metric="cosine")
    drcg.fit_transform(X, y)
    assert hasattr(drcg, "persistence_")


def test_dowker_rips_complex_gudhi_empty_vertices():
    """
    Check whether `DowkerRipsComplexGudhi` runs for empty set of vertices and
    yields correct result.
    """
    X, y = (
        [
            rng.standard_normal(size=(0, 512)),
            rng.standard_normal(size=(10, 512)),
        ],
        None,
    )
    drcg = DowkerRipsComplexGudhi()
    drcg.fit_transform(X, y)
    assert hasattr(drcg, "persistence_")
    assert len(drcg.persistence_) == 2
    assert (drcg.persistence_[0] == np.empty((0, 2))).all()
    assert (drcg.persistence_[1] == np.empty((0, 2))).all()


def test_dowker_rips_complex_gudhi_empty_witnesses():
    """
    Check whether `DowkerRipsComplexGudhi` runs for empty set of witnesses.
    """
    X, y = (
        [
            rng.standard_normal(size=(10, 512)),
            rng.standard_normal(size=(0, 512)),
        ],
        None,
    )
    drcg = DowkerRipsComplexGudhi()
    drcg.fit_transform(X, y)
    assert hasattr(drcg, "persistence_")
    assert len(drcg.persistence_) == 2
    assert (drcg.persistence_[0] == np.empty((0, 2))).all()
    assert (drcg.persistence_[1] == np.empty((0, 2))).all()


def test_dowker_rips_complex_gudhi_empty_witnesses_no_swap():
    """
    Check whether `DowkerRipsComplexGudhi` runs for empty set of witnesses with
    `swap=False`.
    """
    X, y = (
        [
            rng.standard_normal(size=(10, 512)),
            rng.standard_normal(size=(0, 512)),
        ],
        None,
    )
    drcg = DowkerRipsComplexGudhi(swap=False)
    drcg.fit_transform(X, y)
    assert hasattr(drcg, "persistence_")
    assert len(drcg.persistence_) == 2
    assert (drcg.persistence_[0] == np.empty((0, 2))).all()
    assert (drcg.persistence_[1] == np.empty((0, 2))).all()


def test_dowker_rips_complex_gudhi_quadrilateral(quadrilateral):
    """
    Check whether `DowkerRipsComplexGudhi` returns correct result on small
    quadrilateral.
    """
    drcg = DowkerRipsComplexGudhi()
    drcg.fit_transform(*quadrilateral)
    assert hasattr(drcg, "persistence_")
    assert len(drcg.persistence_) == 2
    assert (
        drcg.persistence_[0] == np.array([[1, np.inf]], dtype=np.float64)
    ).all()
    assert (
        drcg.persistence_[1]
        == np.array([[np.sqrt(5), np.sqrt(8)]], dtype=np.float64)
    ).all()


def test_dowker_rips_complex_gudhi_octagon(octagon):
    """
    Check whether `DowkerRipsComplexGudhi` returns correct result on regular
    octagon.
    """
    drcg = DowkerRipsComplexGudhi()
    drcg.fit_transform(*octagon)
    assert hasattr(drcg, "persistence_")
    assert len(drcg.persistence_) == 2
    birth = np.sqrt(2 - np.sqrt(2))
    death = np.sqrt(2 + np.sqrt(2))
    assert (
        drcg.persistence_[0]
        == np.array([[birth, death], [birth, np.inf]], dtype=np.float64)
    ).all()
    assert drcg.persistence_[1].shape == (0, 2)
