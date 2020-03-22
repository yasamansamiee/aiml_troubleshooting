# -*- coding: utf-8 -*-
r"""
Contains the class for incrementally learning vategorical dirichlet process mixture models.
"""

__author__ = "Stefan Ulbrich"
__copyright__ = "Copyright 2018-2020, Acceptto Corporation, All rights reserved."
__credits__ = []
__license__ = "Acceptto Confidential"
__version__ = ""
__maintainer__ = "Stefan Ulbrich"
__email__ = "Stefan.Ulbrich@acceptto.com"
__status__ = "alpha"
__date__ = "2020-03-18"

from typing import Optional
import logging
import numpy as np
import attr

from .base import BaseModel

logger = logging.getLogger(__name__)


@attr.s
class KuberModel(BaseModel):
    """
    A dirichlet mixture model for learning categorical data.
    This covers only a *single* random variable.

    Note a single model does not make much sense for itself. At least
    two features (KuberModels) have to be combined in order to get
    more than one component. Otherwise, only the distribution
    of feature values is estimated. Having more than one feature,
    considers *combinatorics*. The :class:`Kuberspatiotemporal`
    allows for combining children of the abstract :class:`BaseModel`.

    Parameters
    ----------
    n_symbols : int
        [description], defaults to 5

    """

    # public attributes
    n_symbols: int = attr.ib(default=5)  # CATEGORICAL
    # Internal state variable: Probability mass functions
    __pmf: Optional[np.ndarray] = attr.ib(default=None, repr=False)

    def initialize(self):

        logger.info("kuber initialize")
        super().initialize()

        self.n_dim = 1
        self._sufficient_statistics += [np.zeros((self.n_components, self.n_symbols))]

        assert (
            len(self._sufficient_statistics) == 2
        ), f"Warning super method not called (len(S)={len(self._sufficient_statistics)})"

        self.__pmf = np.random.dirichlet([1] * self.n_symbols, self.n_components)
        logger.debug(self.__pmf.shape)

    def reset(self, fancy_index: np.ndarray):
        if self.random_reset:
            self.__pmf[fancy_index] = np.random.dirichlet([1] * self.n_symbols, 1)
        else:
            self.__pmf[fancy_index] = np.zeros(self.n_symbols)

    def expect_components(self, data: np.ndarray) -> np.ndarray:
        assert data.ndim == 2, f"Data should be 2D is {data.ndim}"

        assert data.shape[1] == 1, "Data not a column vector"
        return self.__pmf[:, data.reshape(-1).astype(int)].T

    def maximize_components(self):

        # suppress div by zero warinings (occur naturally for disabled components)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.__pmf = (
                self._sufficient_statistics[1] / self._sufficient_statistics[0][:, np.newaxis]
            )

    def update_statistics(
        # pylint: disable=bad-continuation
        self,
        case: str,
        data: Optional[np.ndarray] = None,
        responsibilities: Optional[np.ndarray] = None,
        rate: Optional[float] = None,
    ):

        assert data.ndim == 2, f"Data should be 2D is {data.ndim}"

        if case == "batch":
            n_samples = data.shape[0]
            temp = np.zeros((n_samples, self.n_components, self.n_symbols))
            temp[np.arange(n_samples), :, data.reshape(-1).astype(int)] = responsibilities
            self._sufficient_statistics[1] = np.sum(temp, axis=0)

        elif case == "online":
            temp = np.zeros((n_samples, self.n_components, self.n_symbols))
            temp[np.arange(n_samples), :, data.astype(int)] = responsibilities
            self._sufficient_statistics[1] = np.sum(temp, axis=0)

        elif case == "init":
            temp = np.zeros((n_samples, self.n_components, self.n_symbols))
            temp[np.arange(n_samples), :, data.astype(int)] = responsibilities
            self._sufficient_statistics[1] = np.sum(temp, axis=0)

    def find_degenerated(self):
        # One could check for a symbol with prob 1 but I don't think that's a good idea
        return np.zeros(self.n_components, dtype=bool)
