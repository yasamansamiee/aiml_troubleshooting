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

import logging
import attr
import numpy as np

from .compound import CompoundModel

logger = logging.getLogger(__name__)




@attr.s
class LazyCompoundModel(CompoundModel):
    """
    Lazy implementation in case of few samples. 
    
    Learning is delayed in favor of a lazy density estimation (cf. 
    `kernel density estimation<https://en.wikipedia.org/wiki/Kernel_density_estimation>`_)
    until enough data has been acquired. After that, learning is triggered.


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

    See Also
    --------

    This class derives from :class:`CompoundModel`.



    """

    min_samples: int = attr.ib(default=0)

    def initialize(self):
        super().initialize()

        if self.min_samples > self.n_components:
            raise ValueError(
                "The minimum of samples for non-lazy learning "
                "must not be larger than the number of components"
                f"{self.min_samples} > {self.n_components}"
            )

    def batch(self, data: np.ndarray, responsibilities: np.ndarray):

        if data.shape[1] >= self.min_samples:
            super().batch(data, responsibilities)
        else:
            self._weights[data.shape[1] :] = 0.0
            self._weights[0 : data.shape[1]] = 1.0 / data.shape[1]

            for feature in self.features:
                feature.model.sync(self._weights, self._sufficient_statistics[0])
                feature.model.lazy_init(data[:, feature.columns])

    def online(
        self, data: np.ndarray, responsibilities: np.ndarray, rate: float,
    ):
        # FIXME implementation missing. Requires a flag that indicates
        # that enough data has been observed

        raise NotImplementedError("Online learning not supported yet")
        # super().online(data, responsibilities, rate)

