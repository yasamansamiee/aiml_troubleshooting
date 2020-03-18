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



import time
from typing import Optional, Tuple
import logging
# from sklearn.datasets import make_spd_matrix
import numpy as np
import attr

from .tools import repr_ndarray, cholesky_precisions
from .base import BaseModel

logger = logging.getLogger(__name__)


# FIXME find_degenerate and reset(_component) will not work currently


# Always useful: https://stackoverflow.com/a/44401529
logging.basicConfig(format='[%(funcName)s:%(lineno)d] %(message)s')
logger = logging.getLogger(__name__)

@attr.s
class SpatialModel(BaseModel):
    # public attributes
    n_dim: int = attr.ib(default=2)
    start_maximimization: int = attr.ib(default=5)
    limits: Tuple[np.ndarray, np.ndarray] = attr.ib(default=None)  # should have a validator
    min_eigval: float = attr.ib(default=1e-2)

    # Internal state variables
    means: Optional[np.ndarray] = attr.ib(default=None, repr=repr_ndarray)
    covs: Optional[np.ndarray] = attr.ib(default=None, repr=repr_ndarray)

    cached_regularization: Optional[np.ndarray] = attr.ib(default=None, repr=False)

    def __attrs_post_init__(self):
        if self.limits is None:
            self.limits = (np.zeros(self.n_dim), np.ones(self.n_dim))

        if self.means is None:
            # (b - a) * random_sample() + a
            self.means = (
                (self.limits[1]-self.limits[0])
                * np.random.random((self.n_components, self.n_dim,))
                + self.limits[0]
            )

        if self.covs is None:
            self.covs = (
                np.tile(np.identity(self.n_dim), (self.n_components, 1, 1))
                * np.random.rand(self.n_components)[:, np.newaxis, np.newaxis]  # (1/self.n_components)
                * 10
            )
        if not self.sufficient_statistics:
            self.sufficient_statistics += [
                np.zeros((self.n_components, self.n_dim)),
                np.zeros((self.n_components, self.n_dim, self.n_dim)),
            ]

        if self.cached_regularization is None:
            self.cached_regularization = np.identity(self.n_dim)*1e-6

    def expect_components(self, data: np.ndarray) -> np.ndarray:

        if data.shape[1] != self.n_dim:
            raise ValueError(f"Wrong input dimensions {data.shape[1]} != {self.n_dim} ")

        # n_samples = data.shape[0]

        # precisions = _compute_precision_cholesky(self.covs,'full')
        # log_det = _compute_log_det_cholesky(precisions,'full', self.n_dim)
        precisions, log_det = cholesky_precisions(self.covs)

        # n - n_samples, d/e - n_dim, c - n_components
        log_prob = np.sum(np.square(
            np.einsum('nd,cde->nce', data, precisions) -
            np.einsum('cd,cde->ce', self.means, precisions)[np.newaxis, :, :]),
            axis=2)

        # weights can / should be zero (caught degenerates)
        with np.errstate(divide='ignore', invalid='ignore'):

            probabilities = np.exp(
                -.5 * (self.n_dim * np.log(2 * np.pi) + log_prob)
                + log_det
            )

        return probabilities

    def update_statistics(self, case: str, data: Optional[np.ndarray]=None, responsibilities: Optional[np.ndarray]=None, rate:Optional[float] = None ):

        if case == 'batch':
            assert data.shape == (self.n_dim)

            self.sufficient_statistics[1] = np.sum(
                responsibilities[:, :, np.newaxis] * data[:, np.newaxis, :],  # (n_samples, n_components, n_dim)
                axis=0
            )

            self.sufficient_statistics[2] = np.sum(
                responsibilities[:, :, np.newaxis, np.newaxis] *
                np.einsum('Ti,Tj->Tij', data, data)[:, np.newaxis, :, :],  # (n_samples, n_components, n_dim, n_dim)
                axis=0
            )
        elif case == 'online':
            assert data.shape == (self.n_dim)

            self.sufficient_statistics[1] += (
                rate * responsibilities[:, np.newaxis] * data[np.newaxis, :]
            )

            self.sufficient_statistics[2] += (
                rate * responsibilities[:, np.newaxis, np.newaxis] *
                np.einsum('i,j->ij', data, data)[np.newaxis, :, :]
            )
        elif case == 'init':
            self.sufficient_statistics[1] = (
                self.sufficient_statistics[0][:, np.newaxis] * self.means
            )

            self.sufficient_statistics[2] = (
                self.sufficient_statistics[0][:, np.newaxis, np.newaxis] *
                (self.covs +
                 np.einsum('ki,kj->kij', self.means, self.means))
            )

    def maximize_components(self):

        # suppress div by zero warinings (occur naturally for disabled components)
        with np.errstate(divide='ignore', invalid='ignore'):

            self.means = self.sufficient_statistics[1] / self.sufficient_statistics[0][:, np.newaxis]

            self.covs = (
                self.sufficient_statistics[2] / self.sufficient_statistics[0][:, np.newaxis, np.newaxis]
                - np.einsum('ki,kj->kij', self.means, self.means)
            ) + self.cached_regularization[np.newaxis, :, :]

    def reset(self,fancy_index: np.ndarray, randomize=False):
        # Rename to reset, and add a parameter for

        # Weights are to be set in detect
        #self.weights[fancy_index] = 1 / self.n_components
        #self.weights /= np.sum(self.weights)

        if randomize:
            self.means[fancy_index] = (
                (self.limits[1]-self.limits[0])
                * np.random.random((self.n_components, self.n_dim,))
                + self.limits[0])[fancy_index]
            self.covs[fancy_index] = np.identity(self.n_dim)
        else:
            pass

    def find_degenerated(self, method='eigen', remove=True):
        '''Remove irrelevant components'''

        if method == 'eigen':
            self.covs[np.any(np.isnan(self.covs), axis=1)] = 0
            irrelevant = np.min(np.linalg.eigvals(self.covs), axis=1) < self.min_eigval

        elif method == 'count':
            # In batch, self.sufficient_statistics is almost equal to self.counter
            # So we can estimate the number of points for each component and remove
            # those with less than enogh points (n+1, e.g., 3 points required in 2D
            # as 2D only spans a line)
            # In online, the original assumption does not hold
            irrelevant = self.weights < (self.n_dim+1) / np.sum(self.sufficient_statistics[0])

        if remove:
            self.priors[irrelevant] = 0.0
            self.means[irrelevant] = 0
            self.covs[irrelevant] = np.identity(self.n_dim) * 1e-2
            # self.pmf[irrelevant] = 0
        else:
            raise NotImplementedError

        if self.nonparametric:
            self.weights = self.stick_breaking()
        else:
            self.weights[irrelevant] = 0.0
            self.weights /= np.sum(self.weights)

        return np.sum(irrelevant)
