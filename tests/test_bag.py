import numpy as np
import pytest
try:
    from sawmil.bag import Bag, BagDataset
except Exception:  # pragma: no cover
    from ..src.sawmil.bag import Bag, BagDataset  # type: ignore


def test_bag_default_mask_and_properties():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    bag = Bag(X=X, y=1.0)
    assert bag.n == 3
    assert bag.d == 2
    assert np.all(bag.mask == 1.0)
    assert np.array_equal(bag.positives(), np.array([0, 1, 2]))
    assert np.array_equal(bag.negatives(), np.array([], dtype=int))


def test_bag_mask_validation_and_clipping():
    X = np.zeros((3, 2))
    with pytest.raises(ValueError):
        Bag(X=X, y=0.0, intra_bag_mask=[1, 0])  # wrong length

    bag = Bag(X=X, y=0.0, intra_bag_mask=[2, 0.2, -1])
    assert np.allclose(bag.mask, [1.0, 0.2, 0.0])
    assert np.array_equal(bag.positives(), np.array([0]))
    assert np.array_equal(bag.negatives(), np.array([1, 2]))


def test_bag_X_not_2D():
    with pytest.raises(ValueError):
        Bag(X=np.array([1, 2, 3]), y=1.0)


@pytest.fixture
def sample_dataset():
    X1 = np.array([[1, 1], [2, 2]], dtype=float)
    X2 = np.array([[3, 3], [4, 4], [5, 5]], dtype=float)
    X3 = np.array([[6, 6]], dtype=float)
    masks = [np.array([1, 0], dtype=float), None, np.array([1], dtype=float)]
    y = [1, 0, 1]
    return BagDataset.from_arrays([X1, X2, X3], y, masks)


def test_bag_dataset_from_arrays_mismatch_lengths():
    Xs = [np.zeros((1, 2)), np.ones((2, 2))]
    ys = [1, 0]
    masks = [np.ones(1)]
    with pytest.raises(ValueError):
        BagDataset.from_arrays(Xs, ys, masks)


def test_bag_dataset_split_and_counts(sample_dataset):
    pos_bags, neg_bags = sample_dataset.split_by_label()
    assert len(pos_bags) == 2
    assert len(neg_bags) == 1
    assert sample_dataset.num_bags == 3
    assert sample_dataset.num_pos_bags == 2
    assert sample_dataset.num_neg_bags == 1
    assert sample_dataset.num_pos_instances == 3
    assert sample_dataset.num_neg_instances == 3
    assert sample_dataset.num_instances == 6
    assert np.array_equal(sample_dataset.y, np.array([1, 0, 1], dtype=float))


def test_bag_dataset_Xy(sample_dataset):
    Xs, ys, masks = sample_dataset.Xy()
    assert len(Xs) == 3
    assert all(X.shape[1] == 2 for X in Xs)
    assert np.array_equal(ys, np.array([1, 0, 1], dtype=float))
    assert np.array_equal(masks[1], np.ones(3))


def test_bag_dataset_positive_negative_instances(sample_dataset):
    X_pos, idx_pos = sample_dataset.positive_instances()
    assert X_pos.shape == (2, 2)
    assert np.array_equal(idx_pos, np.array([0, 2]))
    X_neg, idx_neg = sample_dataset.negative_instances()
    assert X_neg.shape == (4, 2)
    assert np.array_equal(idx_neg, np.array([0, 1, 1, 1]))


def test_bag_dataset_singletons(sample_dataset):
    neg_singletons = sample_dataset.negative_bags_as_singletons()
    assert len(neg_singletons) == 3
    assert all(b.n == 1 and b.y == -1.0 for b in neg_singletons)

    pos_singletons = sample_dataset.positive_bags_as_singletons()
    assert len(pos_singletons) == 3
    assert all(b.n == 1 and b.y == 1.0 for b in pos_singletons)
