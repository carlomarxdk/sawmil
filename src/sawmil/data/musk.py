
from __future__ import annotations
from typing import Optional, Tuple

from ..bag import BagDataset, Bag
import logging
import numpy as np

try:
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required to load Musk.") from exc

log = logging.getLogger('data.musk')
BAG_COL = 'molecule_name'
LABEL_COL = 'class'

def load_musk_bags(
    *,
    version: int = 1,
    test_size: float = 0.2,
    random_state: int = 0,
    standardize: bool = True,
) -> Tuple[BagDataset, BagDataset, StandardScaler | None]:
    """
    Fetch Musk from OpenML, build BagDataset, stratified split by bag label.
    Args:
        version: 'musk' dataset version (default 1)
        test_size: proportion of the dataset to include in the test split (default 0.2)
        random_state: random seed for the sklearn package
        standardize: whether to standardize the features. The StandardScaler will be fit on the training data (default True)
    Returns: (train_ds, test_ds)
    """
    
    musk = fetch_openml(name="musk", version=version, as_frame=True)
    df = musk.frame.copy()

    label = df[LABEL_COL]
    
    bag_list: list[Bag] = []
    cols = df.columns.tolist()
    use_cols = [c for c in cols if c != LABEL_COL and c != BAG_COL]
    for name, grp in df.groupby(df[BAG_COL].astype(str)):
        X = grp[use_cols].to_numpy(dtype=float)
        ys = label.loc[grp.index].to_numpy(dtype=float)
        y_bag = float(ys[0]) if np.all(ys == ys[0]) else float((ys > 0.5).any())
        bag_list.append(Bag(X=X, y=y_bag))  # intra_bag_labels defaults to ones

    # Stratified split at the bag level
    bag_labels = np.array([b.y for b in bag_list], dtype=float)
    idx = np.arange(len(bag_list))
    idx_tr, idx_te = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=bag_labels
    )
    train_bags = [bag_list[i] for i in idx_tr]
    test_bags  = [bag_list[i] for i in idx_te]

    # Standardize features
    if standardize:
        scaler = StandardScaler()
        train_X = np.vstack([b.X for b in train_bags])
        scaler.fit(train_X)
    else:
        scaler = None
    train_bags = [Bag(X=scaler.transform(b.X), y=b.y) for b in train_bags]
    test_bags  = [Bag(X=scaler.transform(b.X), y=b.y) for b in test_bags]

    train_ds = BagDataset(train_bags)
    test_ds  = BagDataset(test_bags)
    return train_ds, test_ds, scaler