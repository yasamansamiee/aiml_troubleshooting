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


from typing import Optional, Tuple, Union
import logging

# from sklearn.datasets import make_spd_matrix
from scipy.stats import multinomial, multivariate_normal

import numpy as np
import attr

from .tools import repr_ndarray, cholesky_precisions
from .base import BaseModel

from .cumulative import boxed_cdf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
print(__name__)

# Black and pylint disagree about line continuation
# pylint: disable=bad-continuation

@attr.s
class SpatialModel(BaseModel):
    """
    Learning Dirichlet process mixture models for spatial data.

    This class implemets nonparametric Gaussian mixture models and
    is not limited to 2D data (i.e., not limited to geospatial data.

    Parameters
    ----------
    BaseModel : [type]
        [description]
    box: float or ndarray
        Defines the box to be used when computing cumulative density 
        function. This parameter might make data processing much slow.
    n_dim : int
        The number of dimensions of the feature space, by default 2
    limits : Optional[Tuple[np.ndarray, np.ndarray]]
        If specified, used to define the lower and upper bounds for
        the random intializations, by default `None`
    min_eigval : float
        Important value. Minimum extend a cluster/component is allowed
        to have in one of its main directions. Prevents degenerated
        components. Read the documentation for details, defaults to
        `1e-2`.
    covar_factor : float
        Defines the initial variances, i.e., the diagonal values for the 
        covariance matrices.
    """

    # public attributes
    limits: Optional[Tuple[np.ndarray, np.ndarray]] = attr.ib(
        default=None
    )  # TODO should have a validator
    min_eigval: float = attr.ib(default=1e-5)
    covar_factor: float = attr.ib(default=0.005)
    box: Optional[Union[float, np.array]] = attr.ib(default=None)

    # Internal state variables
    __means: Optional[np.ndarray] = attr.ib(default=None, repr=repr_ndarray)
    __covs: Optional[np.ndarray] = attr.ib(default=None, repr=repr_ndarray)

    __cached_regularization: Optional[np.ndarray] = attr.ib(default=None, repr=False)

    def initialize(self):

        super().initialize()

        if self.limits is None:
            self.limits = (np.zeros(self.n_dim), np.ones(self.n_dim))

        # (b - a) * random_sample() + a
        self.__means = (self.limits[1] - self.limits[0]) * np.random.random(
            (self.n_components, self.n_dim,)
        ) + self.limits[0]

        self.__covs = (
            np.tile(np.identity(self.n_dim), (self.n_components, 1, 1)) * self.covar_factor
            # * np.random.rand(self.n_components)[:, np.newaxis, np.newaxis]  # (1/self.n_components)
        )
        self._sufficient_statistics += [
            np.zeros((self.n_components, self.n_dim)),
            np.zeros((self.n_components, self.n_dim, self.n_dim)),
        ]

        self.__cached_regularization = np.identity(self.n_dim) * 1e-6

    def expect(self, data: np.ndarray) -> np.ndarray:

        if data.shape[1] != self.n_dim:
            raise ValueError(f"Wrong input dimensions {data.shape[1]} != {self.n_dim} ")

        # n_samples = data.shape[0]

        # precisions = _compute_precision_cholesky(self.covs,'full')
        # log_det = _compute_log_det_cholesky(precisions,'full', self.n_dim)
        precisions, log_det = cholesky_precisions(self.__covs)

        # n - n_samples, d/e - n_dim, c - n_components
        log_prob = np.sum(
            np.square(
                np.einsum("nd,cde->nce", data, precisions)
                - np.einsum("cd,cde->ce", self.__means, precisions)[np.newaxis, :, :]
            ),
            axis=2,
        )

        with np.errstate(divide="ignore", invalid="ignore"):

            probabilities = np.exp(-0.5 * (self.n_dim * np.log(2 * np.pi) + log_prob) + log_det)

        if not self.box is None:

            a = np.array(
                [
                    boxed_cdf(data, self.box, mean, sigma, None, 1e-5, 1e-5)
                    for sigma, mean in zip(self.__covs, self.__means)
                ]
            )
            logger.debug(a.shape)
            return a.T
        else:
            return probabilities

    def batch(self, data: np.ndarray, responsibilities: np.ndarray):
        n_samples = responsibilities.shape[0]
        assert data.shape == (n_samples, self.n_dim)

        self._sufficient_statistics[1] = np.sum(
            responsibilities[:, :, np.newaxis]
            * data[:, np.newaxis, :],  # (n_samples, n_components, n_dim)
            axis=0,
        )

        self._sufficient_statistics[2] = np.sum(
            responsibilities[:, :, np.newaxis, np.newaxis]
            * np.einsum("Ti,Tj->Tij", data, data)[
                :, np.newaxis, :, :
            ],  # (n_samples, n_components, n_dim, n_dim)
            axis=0,
        )

    def online(
        self, data: np.ndarray, responsibilities: np.ndarray, rate: float,
    ):
        assert data.shape == (1, self.n_dim)

        self._sufficient_statistics[1] += (
            rate * responsibilities.reshape(-1)[:, np.newaxis] * data.reshape(-1)[np.newaxis, :]
        )

        self._sufficient_statistics[2] += (
            rate
            * responsibilities.reshape(-1)[:, np.newaxis, np.newaxis]
            * np.einsum("i,j->ij", data.reshape(-1), data.reshape(-1))[np.newaxis, :, :]
        )

    def online_init(self):
        self._sufficient_statistics[1] = (
            self._sufficient_statistics[0][:, np.newaxis] * self.__means
        )

        self._sufficient_statistics[2] = self._sufficient_statistics[0][
            :, np.newaxis, np.newaxis
        ] * (self.__covs + np.einsum("ki,kj->kij", self.__means, self.__means))


    def maximize(self):

        # suppress div by zero warinings (occur naturally for disabled components)
        with np.errstate(divide="ignore", invalid="ignore"):

            self.__means = (
                self._sufficient_statistics[1] / self._sufficient_statistics[0][:, np.newaxis]
            )

            self.__covs = (
                self._sufficient_statistics[2]
                / self._sufficient_statistics[0][:, np.newaxis, np.newaxis]
                - np.einsum("ki,kj->kij", self.__means, self.__means)
            ) + self.__cached_regularization[np.newaxis, :, :]

    def reset(self, fancy_index: np.ndarray):

        if self.random_reset:
            self.__means[fancy_index] = (
                (self.limits[1] - self.limits[0])
                * np.random.random((self.n_components, self.n_dim,))
                + self.limits[0]
            )[fancy_index]
            self.__covs[fancy_index] = np.identity(self.n_dim)
        else:
            self.__means[fancy_index] = 0
            self.__covs[fancy_index] = np.identity(self.n_dim) * 1e-2

    def find_degenerated(self) -> np.ndarray:
        """Remove irrelevant components"""

        # https://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/
        self.__covs[np.any(np.isnan(self.__covs), axis=1)] = 0
        degenerated = np.min(np.linalg.eigvals(self.__covs), axis=1) < self.min_eigval
        return degenerated

    def rvs(self, n_samples: int = 1, idx: Optional[np.ndarray] = None) -> np.ndarray:

        if idx is None:
            idx = multinomial(1, self._weights).rvs(size=n_samples)

        mvn = [
            multivariate_normal(mean=self.__means[i], cov=self.__covs[i],)
            for i in range(self.n_components)
        ]
        rvs = np.array([mvn[i].rvs(size=n_samples) for i in range(self.n_components)])
        rvs = np.swapaxes(rvs, 0, 1)

        return rvs[idx != 0].reshape(-1, self.n_dim)
