# -*- coding: utf-8 -*-
r"""
Provides the class that combines the heterogeneous data estimators.
"""

__author__ = "Stefan Ulbrich"
__copyright__ = "Copyright 2018-2020, Acceptto Corporation, All rights reserved."
__credits__ = []
__license__ = "Acceptto Confidential"
__version__ = ""
__maintainer__ = "Stefan Ulbrich"
__email__ = "Stefan.Ulbrich@acceptto.com"
__status__ = "alpha"
__date__ = "2020-03-19"
__all__ = ["CompoundModel"]

from typing import List, Optional
import logging
import attr
import numpy as np
from scipy import stats

from .base import BaseModel

logger = logging.getLogger(__name__)

# Black and pylint disagree about line continuation
# pylint: disable=bad-continuation


@attr.s
class Feature:
    """Description of features (columns) in the input data."""

    model: BaseModel = attr.ib(repr=lambda x: x.__class__.__name__)
    columns: List[int] = attr.ib(factory=list)
    name: Optional[str] = attr.ib(default=None)


@attr.s
class CompoundModel(BaseModel):
    """
    Composed heterogenous Dirichlet process mixture model.

    Since `sklearn v0.20 <https://scikit-learn.org/stable/whats_new/v0.20.html#highlights>`_ has been released,
    no additional infrastructure is required to operate on Pandas dataframes (unlike PyLoa which has been
    developed prior to that release). The only disadvantage is that the column order of data frames is
    not allowed to change. See below for an example to apply the model.

    Parameters
    ----------
    features : List[Feature]
        Description of the composition

    Example
    -------
    .. code:: python

        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import FunctionTransformer
        from sklearn.pipeline import Pipeline

        import pandas as pd
        d = {'x': [0.5, 0.6], 'y': [0.9, 0.1], 'f2': [1, 2], 'f2': [3, 4]}
        #     x    y  f1  f2
        # 0  0.5  0.9   1   3
        # 1  0.6  0.1   2   4

        pipeline = make_pipeline(
            make_column_transformer(
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "x"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "y"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "f1"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "f2"),
            ),
            CompoundModel(
                n_components=2,
                n_dim=4,
                n_iterations=200,
                scaling_parameter=1.1,
                nonparametric=False,
                online_learning=False,
                score_threshold=ground_truth.score_threshold,  # has been set in the fixture
                features=[
                    Feature(SpatialModel(n_dim=2, n_components=2), [0, 1]),
                    Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [2]),
                    Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [3]),
                ],
            )
        )

        pipeline.fit(d)
        pipeline.score_threshold = pipeline.get_score_threshold(d)
        pipeline.score(d)

    """

    features: List[Feature] = attr.ib(factory=list, repr=False)

    def initialize(self):
        super().initialize()

        for i in self.features:
            i.model.online_learning = self.online_learning
            if not i.model.n_components == self.n_components:
                raise ValueError(
                    f"Features must have the same number of components as"
                    f"the compound: {i.model.n_components} != {self.n_components}"
                )
            # i.model.n_components = self.n_components
            # i.model.initialize()
            i.model.sync(self._weights, self._sufficient_statistics[0])
            # i.model._BaseModel__priors = self._BaseModel__priors.copy()

        # logger.debug(self._sufficient_statistics[0])

    def reset(self, fancy_index: np.ndarray):
        for feature in self.features:
            feature.model.reset(fancy_index)

    def expect(self, data: np.ndarray) -> np.ndarray:
        for feature in self.features:
            feature.model.sync(self._weights, self._sufficient_statistics[0])

        probability = np.prod(
            np.array(
                [
                    feature.model.expect(data[:, feature.columns])
                    for feature in self.features
                ]
            ),
            axis=0,
        )
        assert probability.shape == (
            data.shape[0],
            self.n_components,
        ), f"Wrong shape: {probability.shape}"

        return probability

    def maximize(self):
        for feature in self.features:
            feature.model.sync(self._weights, self._sufficient_statistics[0])
            feature.model.maximize()

    def find_degenerated(self) -> np.ndarray:
        degenerated = False
        for feature in self.features:
            degenerated = degenerated | feature.model.find_degenerated()
            # Comment: `|=` won't broadcast
        return degenerated

    def batch(self, data: np.ndarray, responsibilities: np.ndarray):
        for feature in self.features:
            feature.model.sync(self._weights, self._sufficient_statistics[0])

            feature.model.batch(
                data[:, feature.columns] if data is not None else None, responsibilities
            )

            # Control:
            assert np.allclose(
                self._weights, feature.model._weights
            ), f"\n{self._weights}\n{feature.model._weights}"

    def online_init(self):
        for feature in self.features:
            feature.model.sync(self._weights, self._sufficient_statistics[0])

            feature.model.online_init()

    def online(
        self, data: np.ndarray, responsibilities: np.ndarray, rate: float,
    ):
        for feature in self.features:
            feature.model.sync(self._weights, self._sufficient_statistics[0])

            feature.model.online(
                data[:, feature.columns] if data is not None else None, responsibilities, rate
            )

            # Control:
            assert np.allclose(
                self._weights, feature.model._weights
            ), f"\n{self._weights}\n{feature.model._weights}"

    def rvs(self, n_samples: int = 1, idx: Optional[np.ndarray] = None) -> np.ndarray:

        if idx is None:
            idx = stats.multinomial(1, self._weights).rvs(size=n_samples)

        logger.debug("sum of indices %s", np.sum(idx, axis=0))

        return np.hstack([feat.model.rvs(n_samples, idx) for feat in self.features])
