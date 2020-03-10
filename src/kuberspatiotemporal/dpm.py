# -*- coding: utf-8 -*-
r"""
Contains the class for incrementally learning heterogeneous dirichlet process models.
"""

__author__ = "Stefan Ulbrich"
__copyright__ = "Copyright 2018-2020, Acceptto Corporation, All rights reserved."
__credits__ = []
__license__ = "Acceptto Confidential"
__version__ = ""
__maintainer__ = "Stefan Ulbrich"
__email__ = "Stefan.Ulbrich@acceptto.com"
__status__ = "alpha"
__date__ = "2020-03-10"

from typing import List
import logging
from scipy.stats import multivariate_normal as mvn
from sklearn.datasets import make_spd_matrix
import numpy as np
import attr

from .tools import check_spd

logger = logging.getLogger(__name__)


@attr.s
class DirichletProcessMixtureModel:
    """
    Class representing a Dirichlet process model that can be
    learned in batch and incremental mode. It can handle
    heterogeneous data in the form of

    Parameters
    ----------
        n_components : int
            [description]
        n_dim : int
            [description]
    """

    n_components: int = attr.ib()
    n_dim: int = attr.ib(default=3)

    # TODO consider the model to be numpy arrays instead of lists

    weights: List[float] = attr.ib(factory=list, repr=lambda x: f"`list of length {len(x)}`")
    means: List[np.ndarray] = attr.ib(
        factory=list, repr=lambda x: f"`list of length {len(x)}, elements of shape {np.asarray(x).shape[1:]}`")
    covs: List[np.ndarray] = attr.ib(
        factory=list, repr=lambda x: f"`list of length {len(x)}, elements of shape {np.asarray(x).shape[1:]}`")

    def __attrs_post_init__(self):
        if not self.weights:
            rand = np.random.random((self.n_components,))
            self.weights = (rand / np.sum(rand)).tolist()
        if not self.means:
            self.means = [np.random.random_sample((self.n_dim,))
                          for i in range(self.n_components)]
        if not self.covs:
            self.covs = [make_spd_matrix(self.n_dim)
                         for i in range(self.n_components)]

    def expect(self, data: np.ndarray) -> List[List[float]]:
        """
        Expectation step.

        Returns the responsibilities.
        TODO: Avoid working with loops. But mvn needs to be rewritten


        Parameters
        ----------
        data : np.ndarray
            [description]

        Returns
        -------
        List[List[float]]
            Responsibilities
        """

        responsibilities = []
        for t in range(data.shape[0]):
            denominator = np.sum([self.weights[i] * mvn.pdf(data[t],
                                                            self.means[i],
                                                            self.covs[i])
                                  for i in range(self.n_components)])

            nominators = [self.weights[i] / denominator * mvn.pdf(data[t],
                                                                  self.means[i],
                                                                  self.covs[i])
                          for i in range(self.n_components)]

            responsibilities.append(nominators)

        return responsibilities

    def maximize(self, data: np.ndarray):
        """
        Maximization step.

        Parameters
        ----------
        data : np.ndarray
            [description]
        """

        n_samples = data.shape[0]

        responsibilities = np.asarray(self.expect(data))

        assert responsibilities.shape == (n_samples, self.n_components)

        # responsibilities: n_samples x n_components

        weights = np.sum(responsibilities / n_samples, axis=0)
        self.weights = weights.tolist()

        assert len(self.weights) == self.n_components

        # responsibilities: n_samples x n_components
        # means : n_components x n_dim
        # data: n_samples x n_dim
        # weights: n_components

        logger.debug("Resp. %s", responsibilities.shape)
        logger.debug("Data. %s", data.shape)

        means = (
            np.sum(data[:, np.newaxis, :] * responsibilities[:, :, np.newaxis], axis=0)
            / (weights[:, np.newaxis] * n_samples)
        )

        assert means.shape == (self.n_components, self.n_dim)

        self.means = [i for i in means]



        # Explanation
        # einsum('Xi,Xj->Xij',a,a): outerproduct of each row with itself. For MxN array, leads to MxNxN.
        # np.newaxis increased the dimensionality of the tensor. numpy broadcasting inserts copies along
        # such the new axes to match the other operands dimensionality.
        # Such block operations are very efficient (and easy to read once one is used to it!)

        covs = (
            np.sum(
                np.einsum('Ti,Tj->Tij', data, data)[:, np.newaxis, :, :] *
                responsibilities[:, :, np.newaxis, np.newaxis] /
                (weights[:, np.newaxis, np.newaxis] * n_samples),  # n_samples x n_components x n_dim x n_dim
                axis=0
            ) -
            np.einsum('ki,kj->kij', means, means)  # n_components x n_dim x n_dim
        )

        assert covs.shape == (self.n_components, self.n_dim, self.n_dim)

        self.covs = [i for i in covs]
        for i in self.covs:
            logger.debug("Cov: \n%s\nSPD: %s\n", i, check_spd(i))

    def fit(self, data: np.ndarray, steps=100):
        """
        Batch learning.

        Parameters
        ----------
        data : np.ndarray
            [description]
        steps : int, optional
            [description], by default 100

        Raises
        ------
        ValueError
            Raised if the data has the wrong dimensions.
        """

        if not data.shape[1] == self.n_dim:
            raise ValueError(f"Wrong input dimensions {data.shape[1]} != {self.n_dim} ")
        for i in range(steps):
            logger.debug("Iteration %d", i+1)
            self.maximize(data)

    def fit_recurse(self, data: np.ndarray, steps=100):
        "Incremental learning"
        pass
