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

import numpy as np
import attr

from .tools import repr_list_ndarray

logger = logging.getLogger(__name__)


# FIXME find_degenerate and reset(_component) will not work currently


# Always useful: https://stackoverflow.com/a/44401529
logging.basicConfig(format='[%(funcName)s:%(lineno)d] %(message)s')
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

def repr_ndarray(x: np.ndarray) -> str:
    return f"Array {x.shape}" if x is not None else "None"

@attr.s
class BaseModel(ABC):
    n_components: int = attr.ib()
    nonparametric: bool = attr.ib(default=True)
    scaling_parameter: float = attr.ib(default=2.0)
    alpha: float = attr.ib(default=0.75)


    counter: int = attr.ib(default=0)
    sufficient_statistics: List[np.ndarray] = attr.ib(factory=list, repr=lambda x: repr_list_ndarray)
    priors: Optional[np.ndarray] = attr.ib(default=None, repr=repr_ndarray)
    weights: Optional[np.ndarray] = attr.ib(default=None, repr=repr_ndarray)


    def __attrs_post_init__(self):

        if not self.sufficient_statistics:
            self.sufficient_statistics += [
                np.zeros((self.n_components,)),
            ]

        if self.priors is None:
            self.priors = np.random.beta(1, self.scaling_parameter, (self.n_components,))
            self.priors[-1] = 1.0  # truncate

        if self.weights is None:
            if not self.nonparametric:
                self.weights = np.ones((self.n_components,)) * 1.0 / self.n_components
                print('Noope')
            else:
                self.weights = self.stick_breaking()

    def stick_breaking(self) -> np.ndarray:
        weights = np.empty(self.priors.shape)
        self.priors[-1] = 1.0  # truncate

        weights[0] = self.priors[0]
        weights[1:] = self.priors[1:] * np.cumprod(1 - self.priors)[:-1]

        return weights

    def predict(self, data: np.ndarray) -> np.ndarray:
        responsibilities = np.asarray(self.expect(data))
        return np.argmax(responsibilities, axis=1), responsibilities

    def expect(self, data: np.ndarray) -> np.ndarray:
        weighted_prob = self.expect_components(data) * self.weights[np.newaxis, :]
        responsibilities = weighted_prob / np.sum(weighted_prob, axis=1)[:, np.newaxis]
        responsibilities[np.isnan(responsibilities)] = 0
        return responsibilities

    def maximize(self):

        if not self.nonparametric:
            self.weights = self.sufficient_statistics[0] / np.sum(self.sufficient_statistics[0])
        else:
            #             logger.debug('before: %s', np.sort(self.weights))
            with np.errstate(divide='ignore', invalid='ignore'):

                self.priors = self.sufficient_statistics[0] / (
                    self.scaling_parameter - 1 + np.flip(np.cumsum(np.flip(self.sufficient_statistics[0])))
                )
                self.priors[np.isnan(self.priors)] = 0.0

            self.weights = self.stick_breaking()

        self.maximize_components()

    def batch(self, data: np.ndarray):

        n_samples = data.shape[0]
        self.counter = n_samples

        responsibilities = np.asarray(self.expect(data))

        self.sufficient_statistics[0] = np.sum(
            responsibilities,  # (n_samples, n_components)
            axis=0
        )
        self.update_statistics(case='batch', data=data, responsibilities=responsibilities)

        self.maximize()
        self.find_degenerated()

    def online(self, data: np.ndarray):

        if self.counter == 0:

            self.sufficient_statistics[0] = (
                self.weights * 10
            )

            self.update_statistics(case='init')

        self.counter = 10

        logger.debug("%s, %f,%f,%f", self.sufficient_statistics[0]/np.sum(self.sufficient_statistics[0]),
                     np.sum(self.sufficient_statistics[0]),
                     (self.counter)**(-self.alpha), ((self.counter)**(-self.alpha))**100)

        degs = 0
        for sample in data:

            self.counter += 1

            responsibilities = np.asarray(self.expect(sample.reshape(1, -1))).reshape(-1)

            rate = (self.counter)**(-self.alpha)

            # First reduce the influence of the older samples
            for i in self.sufficient_statistics:
                i *= (1-rate)

            # Then introduce the new sample
            self.sufficient_statistics[0] += (
                rate * responsibilities
            )

            self.update_statistics(case='online', data=sample, responsibilities=responsibilities,rate=rate)

            self.maximize()
            d = self.find_degenerated()
            if d != degs:
                logger.debug("Detected %d degenerated", degs)
                degs = d
        logger.debug("Finished. Detected %d degenerated", degs)

    def fit(self, data: np.ndarray, n_iterations=100, online=False):

        if len(data.shape) != 2:
            raise ValueError(f"Wrong input dimensions (at least 2D)")

        

        if data.shape[0] == 0:
            logger.info("Empty data set")
            return
        logger.info("Learning %d samples (%s)", data.shape[0],
                    "Online" if online else "Batch")

        for i in range(n_iterations):
            if i % 10 == 0:
                logger.info('Step %d/%d', i, n_iterations)
            if online:
                self.online(data)
            else:
                self.batch(data)

    @abstractmethod
    def reset(self,fancy_index: np.ndarray, randomize=False):
        pass

    @abstractmethod
    def expect_components(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def maximize_components(self):
        pass

    @abstractmethod
    def find_degenerated(self, method='eigen', remove=True):
        pass

    @abstractmethod
    def update_statistics(self, case: str, data: Optional[np.ndarray]=None, responsibilities: Optional[np.ndarray]=None, rate:Optional[float] = None ):
        pass
