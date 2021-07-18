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
        Number of possible symbols/values, defaults to 5

    """

    # public attributes
    n_symbols: int = attr.ib(default=5)  # CATEGORICAL

    #: Internal state variable: Probability mass functions, shape (n_components, n_symbols)
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
        # logger.debug(self.__pmf.shape)

    def lazy_init(self, data: np.ndarray):

        if data.shape[1] != self.n_dim:
            raise ValueError(f"Wrong input dimensions {data.shape[1]} != {self.n_dim} ")
        if data.shape[0] > self.n_components:
            raise ValueError(
                f"Too may samples for lazy learning {data.shape[0]} > {self.n_components} "
            )

        self.__pmf[:,:] = 0.0
        self.__pmf[0 : data.shape[0], data.reshape(-1).astype(int)] = 1.0

    def reset(self, fancy_index: np.ndarray):

        if self.random_reset:
            self.__pmf[fancy_index] = np.random.dirichlet([1] * self.n_symbols, 1)
        else:
            self.__pmf[fancy_index] = np.zeros(self.n_symbols)

    def expect(self, data: np.ndarray) -> np.ndarray:

        # FIXME should be exceptions and not assertions
        assert data.ndim == 2, f"Data should be 2D is {data.ndim}"

        assert data.shape[1] == 1, "Data not a column vector"

        if (data >= self.__pmf.shape[1]).any():
            # TODO: why doing these steps (Adriana made them)?
            # If there is a new symbol, it should just return for them 9
            # (hasn't been seen during training)

            n_symbols = int(np.max(data) + 1)
            aux_pmf = np.zeros((self.__pmf.shape[0], n_symbols))
            aux_pmf[:, 0 : self.__pmf.shape[1]] = self.__pmf
            return aux_pmf[:, data.reshape(-1).astype(int)].T
        else:
            return self.__pmf[:, data.reshape(-1).astype(int)].T

    def maximize(self):

        # suppress div by zero warinings (occur naturally for disabled components)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.__pmf = (
                self._sufficient_statistics[1] / self._sufficient_statistics[0][:, np.newaxis]
            )
            # Disabled componentes lead to a zero by zero division.
            # Setting them to zero allows for continued scoring
            self.__pmf[np.isnan(self.__pmf)] = 0.0

            # With online learning, it can happen that probabilities exceed one
            # Sufficient statistics are only approximated
            if self.online_learning:
                if not np.all(self.__pmf <= 1.0):
                    # logger.warning('Probabilities exceed 1.0')
                    self.__pmf = self.__pmf / np.sum(self.__pmf, axis=1)[:, np.newaxis]
                assert np.all(self.__pmf <= 1.0)

    def batch(self, data: np.ndarray, responsibilities: np.ndarray):
        assert data.ndim == 2, f"Data should be 2D is {data.ndim}"
        n_samples = data.shape[0]
        temp = np.zeros((n_samples, self.n_components, self.n_symbols))
        temp[np.arange(n_samples), :, data.reshape(-1).astype(int)] = responsibilities
        self._sufficient_statistics[1] = np.sum(temp, axis=0)

    def online_init(self):
        # P = S1 / S0 => S1 = P * SO
        self._sufficient_statistics[1] = self.__pmf * self._sufficient_statistics[0][:, np.newaxis]

    def online(
        self, data: np.ndarray, responsibilities: np.ndarray, rate: float,
    ):
        assert data.ndim == 2, f"Data should be 2D is {data.ndim}"
        n_samples = data.shape[0]
        temp = np.zeros((n_samples, self.n_components, self.n_symbols))
        temp[np.arange(n_samples), :, data.astype(int)] = responsibilities
        self._sufficient_statistics[1] += rate * np.sum(temp, axis=0)

    def find_degenerated(self):
        # TODO One could check for a symbol with prob 1 but I don't think that's a good idea
        return np.zeros(self.n_components, dtype=bool)

    def rvs(self, n_samples: int = 1, idx: Optional[np.ndarray] = None) -> np.ndarray:

        if idx is None:
            idx = np.random.multinomial(1, self._weights, size=n_samples)

        try:
            feat: np.ndarray = np.array(
                [
                    [np.where(r == 1)[0][0] for r in np.random.multinomial(1, pmf, size=n_samples)]
                    for pmf in self.__pmf
                ]
            ).T
        except ValueError as exeption:
            logger.error("%s, %s", exeption, self.__pmf)

        return feat[idx != 0].reshape(-1, self.n_dim)  # pylint: disable=unsubscriptable-object
