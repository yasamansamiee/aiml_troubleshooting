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
__date__ = "2021-07-17"
__all__ = ["LazyCompoundModel"]

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
    `kernel density estimation <https://en.wikipedia.org/wiki/Kernel_density_estimation>`_)
    until enough data has been acquired. After that, learning is triggered.


    Parameters
    ----------
    min_samples : int
        The minimal number of points for using non-lazy learning.

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

    def fit(self, data: np.ndarray, y=None):
        """See :meth:`sklearn.mixture.GaussianMixture.fit`"""

        assert data.ndim == 2, f"Data should be 2D is {data.ndim}"

        if len(data.shape) != 2:
            raise ValueError("Wrong input dimensions (at least 2D)")

        if data.shape[0] == 0:
            logger.info("Empty data set")
            return

        n_samples = data.shape[0]
        if n_samples >= self.min_samples:
            logger.info("Eager learning on %d>%d samples", n_samples, self.min_samples)
            super().fit(data, y)
        else:

            if self.online_learning:
                raise NotImplementedError("Online learning not supported yet")

            logger.info("Lazy batch learning on %d<%d samples", n_samples, self.min_samples)

            self._weights[n_samples :] = 0.0
            self._weights[0 : n_samples] = 1.0 / n_samples

            for feature in self.features:
                feature.model.sync(self._weights, self._sufficient_statistics[0])
                feature.model.lazy_init(data[:, feature.columns])

        return self
