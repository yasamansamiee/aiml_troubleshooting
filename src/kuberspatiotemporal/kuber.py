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

from .tools import repr_ndarray
from .base import BaseModel

logger = logging.getLogger(__name__)


# FIXME find_degenerate and reset(_component) will not work currently


# Always useful: https://stackoverflow.com/a/44401529
logging.basicConfig(format='[%(funcName)s:%(lineno)d] %(message)s')
logger = logging.getLogger(__name__)


@attr.s
class KuberModel(BaseModel):
    # public attributes
    n_symbols: int = attr.ib(default=5)  # CATEGORICAL
    pmf: Optional[np.ndarray] = attr.ib(default=None, repr=repr_ndarray)

    # Internal state variables
    def __attrs_post_init__(self):
        if not self.sufficient_statistics:
            self.sufficient_statistics += [
                np.zeros((self.n_components, self.n_symbols))
            ]
        if self.pmf is None:
            self.pmf = np.random.dirichlet([1]*self.n_symbols, self.n_components)

    def reset(self, fancy_index: np.ndarray, randomize=False):
        # Rename to reset, and add a parameter for

        # Weights are to be set in detect
        #self.weights[fancy_index] = 1 / self.n_components
        #self.weights /= np.sum(self.weights)

        if randomize:
            pass  # FIXME
        else:
            pass

    def expect_components(self, data: np.ndarray) -> np.ndarray:
        return self.pmf[:, data.astype(int)].T

    def maximize_components(self):

        # suppress div by zero warinings (occur naturally for disabled components)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.pmf = self.sufficient_statistics[3] / self.sufficient_statistics[0][:, np.newaxis]

    def update_statistics(self, case: str, data: Optional[np.ndarray]=None, responsibilities: Optional[np.ndarray]=None, rate:Optional[float] = None ):

        if case == 'batch':
            n_samples = data.shape[0]
            temp = np.zeros((n_samples, self.n_components, self.n_symbols))
            temp[np.arange(n_samples), :, data.astype(int)] = responsibilities
            self.sufficient_statistics[3] = np.sum(temp, axis=0)
        elif case == 'online':
            temp = np.zeros((n_samples, self.n_components, self.n_symbols))
            temp[np.arange(n_samples), :, data.astype(int)] = responsibilities
            self.sufficient_statistics[3] = np.sum(temp, axis=0)
        elif case == 'init':
            temp = np.zeros((n_samples, self.n_components, self.n_symbols))
            temp[np.arange(n_samples), :, data.astype(int)] = responsibilities
            self.sufficient_statistics[3] = np.sum(temp, axis=0)

    def find_degenerated(self, method='eigen', remove=True):
        return None
