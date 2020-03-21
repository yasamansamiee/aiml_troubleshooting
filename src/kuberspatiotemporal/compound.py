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
__all__ = ["KuberspatiotemporalModel"]

from typing import List, Optional
import logging
import attr
import numpy as np

from .base import BaseModel

logger = logging.getLogger(__name__)



@attr.s
class Feature:
    """Description of features (columns) in the input data."""

    model: BaseModel = attr.ib(repr=lambda x: x.__class__.__name__)
    columns: List[int] = attr.ib(factory=list)
    name: Optional[str] = attr.ib(default=None)


@attr.s
class KuberspatiotemporalModel(BaseModel):
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

    .. code::python

        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import FunctionTransformer
        from sklearn.pipeline import Pipeline

        import pandas as pd
        d = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(data=d)
        #    col1  col2
        # 0     1     3
        # 1     2     4
        ct = ColumnTransformer(
            [('fst', FunctionTransformer(lambda x: x*2), ['col1']),
             ('scnd',FunctionTransformer(lambda x: x),['col1','col2'])]
        )
        ct.fit_transform(df)
        # array([[2, 1, 3],
        #        [4, 2, 4]])
        p = Pipeline(
            [
                ("ct", ct),
                (
                    "kst",
                    KuberspatiotemporalModel(
                        features=[Feature(TemporalModel, [1]), Feature(SpatialModel, [2, 3])]
                    ),
                ),
            ]
        )
    """

    features: List[Feature] = attr.ib(factory=list)

    def initialize(self):
        super().initialize()

        for i in self.features:
            if not i.model.n_components == self.n_components:
                raise ValueError(
                    f"Features must have the same number of components as"
                    f"the compound: {i.model.n_components} != {self.n_components}"
                )
            # i.model.n_components = self.n_components
            # i.model.initialize()
            i.model.sync(self._weights)
            # i.model._BaseModel__priors = self._BaseModel__priors.copy()

        logger.debug(self._sufficient_statistics[0])

    def reset(self, fancy_index: np.ndarray):
        for feature in self.features:
            feature.model.reset(fancy_index)

    def expect_components(self, data: np.ndarray) -> np.ndarray:

        probability = np.prod(
            np.array(
                [
                    feature.model.expect_components(data[:, feature.columns])
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

    def maximize_components(self):
        for feature in self.features:
            feature.model.sync(self._weights)
            feature.model.maximize_components()

    def find_degenerated(self) -> np.ndarray:
        degenerated = False
        for feature in self.features:
            degenerated = degenerated | feature.model.find_degenerated()
            # Comment: `|=` won't broadcast
        return degenerated

    def update_statistics(
        # pylint: disable=bad-continuation,unused-argument
        self,
        case: str,
        data: Optional[np.ndarray] = None,
        responsibilities: Optional[np.ndarray] = None,
        rate: Optional[float] = None,
    ):

        super().update_statistics(case, data, responsibilities, rate)

        for feature in self.features:

            feature.model.update_statistics(case, data[:, feature.columns], responsibilities, rate)

            # Control:
            # pylint: disable=protected-access
            assert np.allclose(
                self._sufficient_statistics[0], feature.model._sufficient_statistics[0]
            )
            assert np.allclose(
                self._weights, feature.model._weights
            ), f"\n{self._weights}\n{feature.model._weights}"
