# https://zenn.dev/sunbluesome/articles/0eaa8eea8375dd

import math
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class CombinatorialPurgedCrossValidation:
    """Combinatorial purged cross-validation."""

    n_splits: int = 5
    n_tests: int = 2
    purge_gap: Union[int, pd.Timedelta] = 0
    embargo_gap: Union[int, pd.Timedelta] = 0

    def __post_init__(self) -> None:
        """Post process."""
        if self.n_tests >= self.n_splits:
            raise ValueError("n_tests must be greater than n_splits.")

    @cached_property
    def _test_sets(self) -> List[List[int]]:
        """Return sets of group index for testing."""
        test_sets = []
        for comb in combinations(range(self.n_splits), self.n_tests):
            test_sets.append(list(comb))
        return test_sets

    @cached_property
    def _train_sets(self) -> List[List[int]]:
        """Return sets of group index for training."""
        train_sets = []
        for test_set in self._test_sets:
            train_sets.append(np.setdiff1d(np.arange(self.n_splits), test_set).tolist())
        return train_sets

    @cached_property
    def pathway_labeled(self) -> NDArray[np.integer]:
        """Labeled backtest pathways."""
        n_combs = math.comb(self.n_splits, self.n_tests)

        pathway_flags = np.zeros((n_combs, self.n_splits), bool)
        for i, comb in enumerate(combinations(range(self.n_splits), self.n_tests)):
            pathway_flags[i, comb] = True
        pathway_labeled = pathway_flags.cumsum(axis=0)
        pathway_labeled[~pathway_flags] = 0
        return pathway_labeled

    @cached_property
    def test_set_labels(self) -> List[List[int]]:
        """Return labels of test sets."""
        return [labels[labels > 0].tolist() for labels in self.pathway_labeled]

    def _is_valid_shape(
        self,
        X: Union[NDArray[np.floating], pd.DataFrame],
    ) -> None:
        if X.ndim != 2:
            raise ValueError("X.ndim must be 2.")

    def _is_valid(
        self,
        X: Union[NDArray[np.floating], pd.DataFrame],
    ) -> None:
        if X.ndim != 2:
            raise ValueError("X.ndim must be 2.")

    def _is_valid_gap_purge(self, X: pd.DataFrame) -> None:
        if isinstance(self.purge_gap, int):
            return
        if not isinstance(self.purge_gap, type(X.index[1] - X.index[0])):
            raise ValueError(
                "The type of purge_gap and the type of difference "
                "of index in X must be same."
            )

    def _is_valid_gap_embargo(self, X: pd.DataFrame) -> None:
        if isinstance(self.embargo_gap, int):
            return
        if not isinstance(self.embargo_gap, type(X.index[1] - X.index[0])):
            raise ValueError(
                "The type of embargo_gap and the type of difference "
                "of index in X must be same."
            )

    def purge(self, indices: pd.Index) -> pd.Index:
        if isinstance(self.purge_gap, int):
            return indices[: -self.purge_gap]

        flags = indices <= (indices.max() - self.purge_gap)
        return indices[flags]

    def embargo(self, indices: pd.Index) -> pd.Index:
        if isinstance(self.embargo_gap, int):
            return indices[self.embargo_gap :]
        flags = indices >= (indices.min() + self.embargo_gap)
        return indices[flags]

    def split(
        self,
        X: Union[NDArray[np.floating], pd.DataFrame],
        y: Optional[Union[NDArray[np.floating], pd.DataFrame, pd.Series]] = None,
        return_backtest_labels: bool = False,
    ) -> Iterable:
        """Split data.

        Parameters
        ----------
        X : (N, M) Union[NDArray[np.floating], pd.DataFrame]
            Explanatory variables to split, where N is number of data,
            M is number of features.
        y : (N,) Union[NDArray[np.floating], pd.DataFrame, pd.Series]
            Objective variables to split, where N is number of data.
        return_backtest_labels : bool, by default False.
            If True, return labels test set on backtest path.

        Returns
        -------
        Iterable that generate (Xtrain, ytrain, Xtest, ytest[, labels]) if y was given.
        If y wasn't given, this generates (Xtrain, Xtest[, labels]).
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
        self._is_valid_shape(X)
        self._is_valid_gap_purge(X)
        self._is_valid_gap_embargo(X)

        inds_unique = X.index.unique()
        inds_unique_splitted = np.array_split(inds_unique, self.n_splits)

        for train_gids, test_gids, labels in zip(
            self._train_sets, self._test_sets, self.test_set_labels
        ):
            inds_to_purge = np.array(test_gids) - 1
            inds_to_embargo = np.array(test_gids) + 1

            test_inds_list = [inds_unique_splitted[gid] for gid in test_gids]

            train_inds_list = []
            for gid in train_gids:
                inds = inds_unique_splitted[gid]
                if gid in inds_to_purge:
                    inds = self.purge(inds)
                if gid in inds_to_embargo:
                    inds = self.embargo(inds)
                train_inds_list.append(inds)

            train_inds = np.concatenate(train_inds_list).ravel()

            if y is None:
                if return_backtest_labels:
                    yield (
                        X.loc[train_inds],
                        [X.loc[inds] for inds in test_inds_list],
                        labels,
                    )
                else:
                    yield X.loc[train_inds], [X.loc[inds] for inds in test_inds_list]
            else:
                if return_backtest_labels:
                    yield (
                        X.loc[train_inds],
                        y.loc[train_inds],
                        [X.loc[inds] for inds in test_inds_list],
                        [y.loc[inds] for inds in test_inds_list],
                        labels,
                    )
                else:
                    yield (
                        X.loc[train_inds],
                        y.loc[train_inds],
                        [X.loc[inds] for inds in test_inds_list],
                        [y.loc[inds] for inds in test_inds_list],
                    )
