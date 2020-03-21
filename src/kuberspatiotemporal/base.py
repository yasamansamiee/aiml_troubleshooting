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

import logging
from abc import ABC, abstractmethod
from typing import Optional, List
from sklearn.base import BaseEstimator, DensityMixin

import numpy as np
import attr

from .tools import repr_list_ndarray, repr_ndarray

logger = logging.getLogger(__name__)

@attr.s
class BaseModel(DensityMixin, BaseEstimator, ABC):
    r"""
    Base class for the Dirichlet process mixture models in this package.
    Learning works in batch and incremental mode.
    Parameters
    ----------
        n_components : int
            Number of mixture components, by default 100
        alpha: float
            Forgetting factor :math:`0.5<\alpha\leg1`, by default 0.5
        nonparametric : bool
            Switch between nonparametric Dirichlet process model and regular
            finite mixture model, by default True
        scaling_parameter : float
            The scaling parameter of the Dirichlet process, by default 2.0
        random_reset : bool
            Reasign irrelevant components with random values,
            by default: False
    """
    n_dim: int = attr.ib(default=2)
    n_components: int = attr.ib(default=100)
    nonparametric: bool = attr.ib(default=True)
    scaling_parameter: float = attr.ib(default=2.0)
    alpha: float = attr.ib(default=0.75)
    random_reset: bool = attr.ib(default=False)

    counter: int = attr.ib(default=0)
    _sufficient_statistics: List[np.ndarray] = attr.ib(
        factory=list, repr=lambda x: repr_list_ndarray
    )
    __priors: Optional[np.ndarray] = attr.ib(default=None, repr=repr_ndarray)
    _weights: Optional[np.ndarray] = attr.ib(default=None, repr=repr_ndarray)

    def __attrs_post_init__(self):
        logger.warning("in __attrs_post_init__")
        self.initialize()

    def initialize(self):
        logger.info('base initialize')

        self._sufficient_statistics += [
            np.zeros((self.n_components,)),
        ]

        self.__priors = np.random.beta(1, self.scaling_parameter, (self.n_components,))
        self.__priors[-1] = 1.0  # truncate

        if not self.nonparametric:
            self._weights = np.ones((self.n_components,)) * 1.0 / self.n_components
        else:
            self._weights = self.stick_breaking()

    def sync(self, weights: np.ndarray):
        """
        Sync estimators in a compound.j
        
        Parameters
        ----------
        weights : np.ndarray
            Weights are protected and therefore be shared.
        """
        self._weights = weights # not necessary to copy


    def stick_breaking(self) -> np.ndarray:
        """
        Implementation of the Dirichlet process's stick breaking
        definition.

        Returns
        -------
        np.ndarray
            [description]
        """
        _weights = np.empty(self.__priors.shape)
        self.__priors[-1] = 1.0  # truncate

        _weights[0] = self.__priors[0]
        _weights[1:] = self.__priors[1:] * np.cumprod(1 - self.__priors)[:-1]

        return _weights

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        [summary]

        Parameters
        ----------
        data : np.ndarray
            [description]

        Returns
        -------
        np.ndarray
            [description]
        """
        responsibilities = np.asarray(self.expect(data))
        return np.argmax(responsibilities, axis=1), responsibilities

    def expect(self, data: np.ndarray) -> np.ndarray:
        """
        Expectation step

        Parameters
        ----------
        data : np.ndarray
            [description]

        Returns
        ------- 
        np.ndarray
            [description]
        """
        weighted_prob = self.expect_components(data) * self._weights[np.newaxis, :]
        responsibilities = weighted_prob / np.sum(weighted_prob, axis=1)[:, np.newaxis]
        # logger.warning('NaN in responsibilities %s', np.sum(np.isnan(responsibilities)))
        responsibilities[np.isnan(responsibilities)] = 0
        logger.debug("%s %s", weighted_prob.shape, responsibilities.shape)
        assert responsibilities.shape == (data.shape[0], self.n_components), f"Wrong shape: {responsibilities.shape}"
        return responsibilities

    def maximize(self):
        """
        Maximization step
        """

        if not self.nonparametric:
            self._weights = self._sufficient_statistics[0] / np.sum(self._sufficient_statistics[0])
        else:
            #             logger.debug('before: %s', np.sort(self._weights))
            with np.errstate(divide="ignore", invalid="ignore"):

                self.__priors = self._sufficient_statistics[0] / (
                    self.scaling_parameter
                    - 1
                    + np.flip(np.cumsum(np.flip(self._sufficient_statistics[0])))
                )
                self.__priors[np.isnan(self.__priors)] = 0.0

            self._weights = self.stick_breaking()

        self.maximize_components()

    def batch(self, data: np.ndarray):
        """
        Batch learning

        Parameters
        ----------
        data : np.ndarray
            [description]
        """

        n_samples = data.shape[0]

        if not data.shape[1] == self.n_dim:
            raise ValueError(f"Wrong number input dimensions: {data.shape[1]} != {self.n_dim}")

        self.counter = n_samples

        responsibilities = np.asarray(self.expect(data))

        self.update_statistics(case="batch", data=data, responsibilities=responsibilities)

        self.maximize()
        self.find_degenerated()

    def online(self, data: np.ndarray):
        """
        Online learning

        Parameters
        ----------
        data : np.ndarray
            [description]
        """

        if self.counter == 0:

            self._sufficient_statistics[0] = self._weights * 10

            self.update_statistics(case="init")

        self.counter = 10

        logger.debug(
            "%s, %f,%f,%f",
            self._sufficient_statistics[0] / np.sum(self._sufficient_statistics[0]),
            np.sum(self._sufficient_statistics[0]),
            (self.counter) ** (-self.alpha),
            ((self.counter) ** (-self.alpha)) ** 100,
        )

        degs = 0
        for sample in data:

            self.counter += 1

            responsibilities = np.asarray(self.expect(sample.reshape(1, -1))).reshape(-1)

            rate = (self.counter) ** (-self.alpha)

            # First reduce the influence of the older samples
            for i in self._sufficient_statistics:
                i *= 1 - rate

            # Then introduce the new sample
            self._sufficient_statistics[0] += rate * responsibilities

            self.update_statistics(
                case="online", data=sample, responsibilities=responsibilities, rate=rate
            )

            self.maximize()
            d = self.find_degenerated()
            if d != degs:
                logger.debug("Detected %d degenerated", degs)
                degs = d
        logger.debug("Finished. Detected %d degenerated", degs)

    def fit(self, data: np.ndarray, n_iterations=100, online=False):
        """
        [summary]

        Parameters
        ----------
        data : np.ndarray
            [description]
        n_iterations : int, optional
            [description], by default 100
        online : bool, optional
            [description], by default False
        """

        assert data.ndim == 2, f"Data should be 2D is {data.ndim}"


        if len(data.shape) != 2:
            raise ValueError(f"Wrong input dimensions (at least 2D)")

        if data.shape[0] == 0:
            logger.info("Empty data set")
            return
        logger.info("Learning %d samples (%s)", data.shape[0], "Online" if online else "Batch")

        for i in range(n_iterations):
            if i % 10 == 0:
                logger.info("Step %d/%d", i, n_iterations)
            if online:
                self.online(data)
            else:
                self.batch(data)

    def update_statistics(
        # pylint: disable=bad-continuation,unused-argument
        self,
        case: str,
        data: Optional[np.ndarray] = None,
        responsibilities: Optional[np.ndarray] = None,
        rate: Optional[float] = None,
    ):
        """
        Update the sufficient statistics required for online
        learning and batch learning.

        Subclasses should call the method of the Base class
        with `super()` to update the first entry of the
        sufficient statistics.

        Parameters
        ----------
        case : str
            'batch', 'online', 'init'
        data : Optional[np.ndarray], optional
            [description], by default None
        responsibilities : Optional[np.ndarray], optional
            [description], by default None
        rate : Optional[float], optional
            [description], by default None
        """


        if case == "batch":
            self._sufficient_statistics[0] = np.sum(
                responsibilities,  # (n_samples, n_components)
                axis=0
            )

        elif case == "online":
            self._sufficient_statistics[0] += (
                rate * responsibilities
            )

        elif case == "init": # actually init could go to __attrs_post_init__
            self._sufficient_statistics[0] = (
                self._weights * 10
            )

    def score(self, X: np.ndarray, y=None)->float:
        """
        TODO: Fill out

        Parameters
        ----------
        X : np.ndarray
            [description]
        y : [type], optional
            [description], by default None

        Returns
        -------
        float
            [description]
        """
        # FIXME: implement
        return 0



    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return individual propabilities of the components
        TODO: Check whether this is expect_components

        Parameters
        ----------
        X : np.ndarray
            The data for predicting the propabilities, shape: (n_samples, n_dim)

        Returns
        -------
        np.ndarray
            The individual propabilities, shape: (n_samples, n_components)
        """
        # FIXME: implement

    # @abstractmethod
    def reset(self, fancy_index: np.ndarray):
        """
        Resets components found by :meth:`find_degenerated`.

        This function and :meth:`find_degenerated` have to
        be separated ad degenerated from all features have
        to be collected/merged prior to reset.

        Parameters
        ----------
        fancy_index : np.ndarray
            Returned by :meth:`find_degenerated`. Used for indexing
            array attributes.
        """
        if self.nonparametric:
            self.__priors[fancy_index] = 0.0
            self._weights = self.stick_breaking()
        else:
            self._weights[fancy_index] = 0.0
            self._weights /= np.sum(self._weights)

    @abstractmethod
    def expect_components(self, data: np.ndarray) -> np.ndarray:
        """Check whether same as prob_a"""

    @abstractmethod
    def maximize_components(self):
        """Maximization step for the individual components"""

    @abstractmethod
    def find_degenerated(self) -> np.ndarray:
        """Select components that are degenerated and need to be reset"""
