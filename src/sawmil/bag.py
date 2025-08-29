# mil/bag.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional
import numpy as np
import numpy.typing as npt

Label = float  # or int; we'll store 0/1 or -1/+1; convert as needed downstream

@dataclass
class Bag:
    """A bag of instances with a bag-level label and per-instance (0/1) flags."""
    X: npt.NDArray[np.float64]            # shape (n_i, d)
    y: Label                              # bag label (e.g., 0/1 or -1/+1)
    intra_bag_label: Optional[npt.NDArray[np.float64]] = None  # shape (n_i,), 0/1

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        if self.X.ndim != 2:
            raise ValueError("Bag.X must be 2D (n_i, d).")
        n_i = self.X.shape[0]
        if self.intra_bag_label is None:
            # default: all ones
            self.intra_bag_label = np.ones(n_i, dtype=float)
        else:
            self.intra_bag_label = np.asarray(self.intra_bag_label, dtype=float).ravel()
            if self.intra_bag_label.shape[0] != n_i:
                raise ValueError("intra_bag_label length must match number of instances.")

    @property
    def n(self) -> int: return self.X.shape[0]

    @property
    def d(self) -> int: return self.X.shape[1]

    @property
    def mask(self) -> npt.NDArray[np.float64]:
        """Safe 0/1 mask (clips negatives and >1)."""
        return np.clip(self.intra_bag_label, 0.0, 1.0)

    def positives(self) -> npt.NDArray[np.int64]:
        """Indices of instances with intra_bag_label == 1."""
        return np.flatnonzero(self.mask >= 0.5)

    def negatives(self) -> npt.NDArray[np.int64]:
        """Indices of instances with intra_bag_label == 0."""
        return np.flatnonzero(self.mask < 0.5)

@dataclass
class BagDataset:
    bags: List[Bag]

    @staticmethod
    def from_arrays(
        bags: Sequence[np.ndarray],
        y: Sequence[float],
        intra_bag_labels: Sequence[np.ndarray] | None = None
    ) -> "BagDataset":
        if intra_bag_labels is None:
            intra_bag_labels = [None] * len(bags)
        if len(bags) != len(y) or len(bags) != len(intra_bag_labels):
            raise ValueError("bags, y, intra_bag_labels must have same length.")
        return BagDataset([
            Bag(X=b, y=float(lbl), intra_bag_label=ibl)
            for b, lbl, ibl in zip(bags, y, intra_bag_labels)
        ])

    def split_by_label(self) -> tuple[list[Bag], list[Bag]]:
        return self.positive_bags(), self.negative_bags()

    def Xy(self) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
        Xs = [b.X for b in self.bags]
        ys = np.asarray([b.y for b in self.bags], dtype=float)
        masks = [b.mask for b in self.bags]
        return Xs, ys, masks

    def positive_bags(self) -> list[Bag]:
        return [b for b in self.bags if float(b.y) > 0.0]

    def negative_bags(self) -> list[Bag]:
        return [b for b in self.bags if float(b.y) <= 0.0]

    def positive_instances(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Instances from positive bags with intra_bag_label == 1.
        Returns:
            X_pos: (N, d)
            bag_index: (N,) indices into self.bags (original positions)
        """
        Xs, bag_idx = [], []
        for i, b in enumerate(self.bags):          # iterate original list!
            if float(b.y) <= 0.0:
                continue
            mask = b.mask >= 0.5
            if np.any(mask):
                Xs.append(b.X[mask])
                bag_idx.extend([i] * int(mask.sum()))
        if not Xs:
            d = self.bags[0].d if self.bags else 0
            return np.zeros((0, d)), np.array([], dtype=int)
        return np.vstack(Xs), np.array(bag_idx, dtype=int)

    def negative_instances(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Instances from:
        - all negative bags (all instances),
        - plus from positive bags where intra_bag_label == 0.
        Returns:
            X_neg: (M, d)
            bag_index: (M,) indices into self.bags (original positions)
        """
        Xs, bag_idx = [], []
        for i, b in enumerate(self.bags):          # iterate original list!
            if float(b.y) <= 0.0:
                # take all instances
                if b.X.shape[0] > 0:
                    Xs.append(b.X)
                    bag_idx.extend([i] * b.X.shape[0])
            else:
                # positive bag: only intra == 0
                mask = b.mask < 0.5
                if np.any(mask):
                    Xs.append(b.X[mask])
                    bag_idx.extend([i] * int(mask.sum()))
        if not Xs:
            d = self.bags[0].d if self.bags else 0
            return np.zeros((0, d)), np.array([], dtype=int)
        return np.vstack(Xs), np.array(bag_idx, dtype=int)

    def negative_bags_as_singletons(self) -> list[Bag]:
        '''
        Transforms all negative bags into singleton bags, by flattening each bag  (b, n, d) -> (b x n, d)
        '''
        singletons: list[Bag] = []
        for b in self.negative_bags():
            for j in range(b.n):
                singletons.append(Bag(X=b.X[j:j+1, :], y=-1.0))
        return singletons

    def positive_bags_as_singletons(self) -> list[Bag]:
        '''
        Transforms all positive bags into singleton bags, by flattening each bag  (b, n, d) -> (b x n, d)
        '''
        singletons: list[Bag] = []
        for b in self.positive_bags():
            for j in range(b.n):
                singletons.append(Bag(X=b.X[j:j+1, :], y=1.0))
        return singletons

    @property
    def num_pos_instances(self) -> int:
        return sum(b.n for b in self.positive_bags())

    @property
    def num_neg_instances(self) -> int:
        return sum(b.n for b in self.negative_bags())

    @property
    def num_instances(self) -> int:
        return self.num_pos_instances + self.num_neg_instances

    @property
    def num_bags(self) -> int:
        return len(self.bags)
    
    @property
    def num_pos_bags(self) -> int:
        return len(self.positive_bags())

    @property
    def num_neg_bags(self) -> int:
        return len(self.negative_bags())