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

from .tools import check_spd, check_singular, repr_list_ndarray

logger = logging.getLogger(__name__)


@attr.s
class DirichletProcessMixtureModel:
    r"""
    Class representing a Dirichlet process model that can be
    learned in batch and incremental mode. It can handle
    heterogeneous data in the form of

    Parameters
    ----------
        n_components : int
            Number of mixture components, by default 3
        n_dim : int
            [description]
        alpha: float
            Forgetting factor :math:`0.5<\alpha\leg1`, by default 0.5
        start_maximimization: int
            The number of samples to be processed before online
            learning starts maximization (otherwise learning fails),
            by default 5
    """

    # public attributes
    n_components: int = attr.ib()
    n_dim: int = attr.ib(default=3)
    alpha: float = attr.ib(default=0.5)
    start_maximimization: int = attr.ib(default=5)

    # Internal state variables
    counter: int = attr.ib(default=0)  # This is intended for resuming
    # TODO consider the model to be numpy arrays instead of lists
    weights: List[float] = attr.ib(factory=list, repr=lambda x: f"`list of length {len(x)}`")
    means: List[np.ndarray] = attr.ib(factory=list, repr=lambda x: repr_list_ndarray)
    covs: List[np.ndarray] = attr.ib(factory=list, repr=lambda x: repr_list_ndarray)
    sufficient_statistics: List[np.ndarray] = attr.ib(factory=list, repr=lambda x: repr_list_ndarray)

    def __attrs_post_init__(self):
        if not self.weights:
            rand = np.random.random((self.n_components,))
            self.weights = (rand / np.sum(rand)).tolist()
        if not self.means:
            self.means = [np.random.random_sample((self.n_dim,))
                          for i in range(self.n_components)]
        if not self.covs:
            self.covs = [make_spd_matrix(self.n_dim) * 0.01
                         for i in range(self.n_components)]
        if not self.sufficient_statistics:
            self.sufficient_statistics = [
                np.zeros((self.n_components,)),
                np.zeros((self.n_components, self.n_dim)),
                np.zeros((self.n_components, self.n_dim, self.n_dim))
            ]

    # TODO Avoid working with loops. But mvn needs to be rewritten

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

        for i in self.covs:
            if check_singular(i, 1e11):
                logger.error('Found singular matrix')
                # raise ArithmeticError(f"Singular covariance matrix:\n{i}")
                # TODO think about how to handle this case.
                # The covariance / mean could be randomly reassigned.

        responsibilities = []
        for t in range(data.shape[0]):
            try:
                denominator = np.sum([self.weights[i] * mvn.pdf(data[t],
                                                                self.means[i],
                                                                self.covs[i])
                                      for i in range(self.n_components)])

                nominators = [self.weights[i] / denominator
                              * mvn.pdf(data[t],
                                        self.means[i],
                                        self.covs[i])
                              for i in range(self.n_components)]
            except np.linalg.LinAlgError as err:
                logger.exception("Singular matrix:\n%s,%s",
                                 err, [np.linalg.cond(x) for x in self.covs])

            responsibilities.append(nominators)

        return responsibilities

    def maximize(self):
        """
        Maximization step.

        The maximization step operates on
        the sufficient statistics instead of the data.
        """

        weights = self.sufficient_statistics[0] / np.sum(self.sufficient_statistics[0])
        means = self.sufficient_statistics[1] / self.sufficient_statistics[0][:, np.newaxis]
        covs = (
            self.sufficient_statistics[2] / self.sufficient_statistics[0][:, np.newaxis, np.newaxis]
            - np.einsum('ki,kj->kij', means, means)
        )

        # assert weights.shape == (self.n_components,)
        # assert means.shape == (self.n_components, self.n_dim)
        # assert covs.shape == (self.n_components, self.n_dim, self.n_dim)

        self.weights = weights.tolist()
        self.means = [i for i in means]
        self.covs = [i for i in covs]

        for i, cov in enumerate(self.covs):
            if not check_spd(cov):
                logger.info("Sufficient statistics for component %d: %s", i,
                            [j[i] for j in self.sufficient_statistics])
                raise ArithmeticError("Encountered non-SPD matrix")

    def batch(self, data: np.ndarray):
        """
        Batch learning.

        Parameters
        ----------
        data : np.ndarray
            , shape (n_samples, n_dim)
        """

        # Compute the sufficient statistics
        # Instead of regular EM. Then the maximization is the same for batch and incremental

        n_samples = data.shape[0]

        responsibilities = np.asarray(self.expect(data))
        # assert responsibilities.shape == (n_samples, self.n_components)

        self.sufficient_statistics[0] = np.sum(
            responsibilities,  # (n_samples, n_components)
            axis=0
        )

        self.sufficient_statistics[1] = np.sum(
            responsibilities[:, :, np.newaxis] * data[:, np.newaxis, :],  # (n_samples, n_components, n_dim)
            axis=0
        )

        self.sufficient_statistics[2] = np.sum(
            responsibilities[:, :, np.newaxis, np.newaxis] *
            np.einsum('Ti,Tj->Tij', data, data)[:, np.newaxis, :, :],  # (n_samples, n_components, n_dim, n_dim)
            axis=0
        )

        # assert self.sufficient_statistics[0].shape == (self.n_components,)
        # assert self.sufficient_statistics[1].shape == (self.n_components, self.n_dim)
        # assert self.sufficient_statistics[2].shape == (self.n_components, self.n_dim, self.n_dim)

        self.maximize()

        self.counter = n_samples

    def incremental(self, data: np.ndarray):
        """
        Batch learning.

        Parameters
        ----------
        data : np.ndarray
            , shape (1, n_dim)
        """

        for sample in data:

            self.counter += 1
            # logger.debug('Observing sample %d\n%s', self.counter, sample)

            responsibilities = np.asarray(self.expect(sample.reshape(1, -1))).reshape(-1)

            # assert responsibilities.shape == (self.n_components,)

            rate = (self.counter+1)**(-self.alpha)
            # logger.debug('Rate at iteration %d: %f', counter, rate)

            # First reduce the influence of the older samples
            for i in self.sufficient_statistics:
                i *= (1-rate)

            # Then introduce the new sample
            self.sufficient_statistics[0] += (
                rate * responsibilities
            )

            self.sufficient_statistics[1] += (
                rate * responsibilities[:, np.newaxis] * sample[np.newaxis, :]
            )

            self.sufficient_statistics[2] += (
                rate * responsibilities[:, np.newaxis, np.newaxis] *
                np.einsum('i,j->ij', sample, sample)[np.newaxis, :, :]
            )

            # assert self.sufficient_statistics[0].shape == (self.n_components,)
            # assert self.sufficient_statistics[1].shape == (self.n_components, self.n_dim)
            # assert self.sufficient_statistics[2].shape == (self.n_components, self.n_dim, self.n_dim)

            if self.counter > self.start_maximimization:
                self.maximize()

    def fit(self, data: np.ndarray, n_iterations=100, online=False):
        """
        Start the learning process.

        Parameters
        ----------
        data : np.ndarray
            The samples to be learned (row matrix).
            Array needs to be 2D. Shape: (n_samples, n_components)
        n_iterations : int, optional
            Either EM iterations (if online is False)
            or mini batch (if online is True), by default 100
        online : bool, optional
            Online learning (True) or batch learning (False),
            by default False

        Raises
        ------
        ValueError
            Raised when data does not have the required dimensions
        """
        if len(data.shape) != 2:
            raise ValueError(f"Wrong input dimensions (at least 2D)")

        if data.shape[1] != self.n_dim:
            raise ValueError(f"Wrong input dimensions {data.shape[1]} != {self.n_dim} ")

        logger.debug("Learning %d samples (%s)", data.shape[0],
                     "Online" if online else "Batch")

        for i in range(n_iterations):
            if i % 10 == 0:
                logger.debug('Step %d/%d', i, n_iterations)
            if online:
                self.incremental(data)
            else:
                self.batch(data)
