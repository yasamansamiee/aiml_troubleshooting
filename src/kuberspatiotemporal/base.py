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
from typing import Optional, List, Tuple
from sklearn.base import BaseEstimator, DensityMixin

# from sklearn.mixture import _base, _gaussian_mixture

import numpy as np
import attr

from .tools import repr_list_ndarray, repr_ndarray

logger = logging.getLogger(__name__)

# Black and pylint disagree about line continuation
# pylint: disable=bad-continuation


@attr.s
class BaseModel(DensityMixin, BaseEstimator, ABC):
    r"""
    Base class for the Dirichlet process mixture models in this package.
    Learning works in batch and incremental mode.



    Parameters
    ----------
        n_dim : int
            Number of output dimensions, by default 2
        online_learning : bool
            Switch between online and batch learning, by default :code:`False`
        score_threshold : float or None
            Threshold for assigning considering a sample to be generated
            by the mixture distribution. Use :meth:`get_score_threshold`
            to derive this value. If :code:`None`, return logs of the probabilities (densities or masses).
            By default None
        n_components : int
            Number of mixture components, by default 100
        decay: float
            Forgetting factor :math:`0.5<\alpha_{\text{online}}\leq 1`, by default 0.5
        nonparametric : bool
            Switch between nonparametric Dirichlet process model and regular
            finite mixture model, by default :code:`True`
        scaling_parameter : float
            The scaling parameter :math:`\alpha` of the Dirichlet process, by default 2.0
        random_reset : bool
            Reasign irrelevant components with random values, TODO: Not tested,
            by default: False
        loa: bool
            see :meth:`compute_loa`, by default: False
        n_iterations : int
            In case of batch learning, this number refer to the maximal number of
            alternations in the EM algorith. In online learning, this parameter is
            used to set how many times the same data set should be processed (mini batch),
            TODO: Incremental learning should be revised.
            by default 100
    """

    ###############################################################################################
    # Public attributes (should be set during initialization) (document in class documentation)
    n_dim: int = attr.ib(default=2)

    n_components: int = attr.ib(default=100)

    nonparametric: bool = attr.ib(default=True)

    scaling_parameter: float = attr.ib(default=2.0)

    decay: float = attr.ib(default=0.75)

    online_learning: bool = attr.ib(default=False)

    n_iterations: int = attr.ib(default=100)

    score_threshold: Optional[float] = attr.ib(default=None)

    random_reset: bool = attr.ib(default=False)

    loa: bool = attr.ib(default=False)

    ###############################################################################################
    # Private & protected / internal attributes (document inline)

    #: Sufficient statistics used for computing parameters in the maximization step
    #: The base class defines the first element. Additionally required statistics
    #: have to be implemented in the subclasses.
    #:
    #: .. math::
    #:
    #:     S_0 = \left( \sum_{t\in T} \mathbb E[\delta_i(x)|y_{T},\Phi] \right)_{i \in I}
    _sufficient_statistics: List[np.ndarray] = attr.ib(factory=list, repr=repr_list_ndarray)

    #: Ratios :math:`\nu_i \sim \text{Beta}(1,\alpha)`
    #: for the stick-breaking process (if :attr:`nonparametric` is :code:`True`)
    __priors: Optional[np.ndarray] = attr.ib(default=None, repr=repr_ndarray)

    #: Weights :math:`\pi_i` of the mixture
    _weights: Optional[np.ndarray] = attr.ib(default=None, repr=repr_ndarray)

    #: Counts how many samples have been processed (mainly used for online learning)
    __counter: int = attr.ib(default=0, repr=False)

    ###############################################################################################
    # Public methods
    ###############################################################################################

    def get_score_threshold(self, data: np.ndarray, quantile=0.05):
        """
        Computes a threshold based on a trained model.

        Determines the threshold required to classify 1-:code:`quantile`
        of the given data as belonging to this distribution.

        Parameters
        ----------
        model : BaseEstimator
            [description]
        data : np.ndarray
            [description]
        quantile : float, optional
            [description], by default 0.95
        """
        return np.quantile(self.__expect(data)[1], quantile)

    def sync(self, w0_: np.ndarray, s0_: np.ndarray):
        """
        Sync estimators in a compound.

        Parameter names are chosen so they don't give insight
        after minification.

        Parameters
        ----------
        w0_ : np.ndarray
            Weights are protected and therefore should not be shared directly.
        s0_:
            First sufficient statistics
        """
        self._weights = w0_  # not necessary to copy
        self._sufficient_statistics[0] = s0_

    ###############################################################################################
    # Private methods
    ###############################################################################################

    def __attrs_post_init__(self):
        """
        Hook from the attrs package delegates to custom
        :meth:`__initialize()` which calls overriden/abstract :meth:`initialize()`
        """

        # logger.warning("in __attrs_post_init__")
        self.__initialize()

    def __initialize(self):
        """
        Initialize the object. Calls abstract :meth:`initialize`
        """

        self._sufficient_statistics += [
            np.zeros((self.n_components,)),
        ]

        self.__priors = np.random.beta(1, self.scaling_parameter, (self.n_components,))
        self.__priors[-1] = 1.0  # truncate

        if not self.nonparametric:
            self._weights = np.random.random(self.n_components)
            self._weights /= np.sum(self._weights)
            logger.debug(self._weights)
            logger.debug(np.sum(self._weights))
            # self._weights = np.ones((self.n_components,)) * 1.0 / self.n_components
        else:
            self._weights = self.__stick_breaking()

        self.initialize()

    def __stick_breaking(self) -> np.ndarray:
        """
        Implementation of the Dirichlet process's stick breaking
        definition.

        Returns
        -------
        weights: np.ndarray, shape (n_components,)
            New weights.
        """
        weights = np.empty(self.__priors.shape)
        self.__priors[-1] = 1.0  # truncate

        weights[0] = self.__priors[0]
        weights[1:] = self.__priors[1:] * np.cumprod(1 - self.__priors)[:-1]

        return weights

    def __expect(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Expectation step

        Returns three arrays:

        * **responsibilities** : np.ndarray, shape (n_samples, n_components)

            .. math::

                \left(\frac{\pi_i P(x_t=i|y_t,\Phi)}{\sum_{j\in I} \pi_j P(x_t=j|y_t,\Phi)}\right)_{i \in I, t\in T}

        * *score_samples** : np.ndarray, shape (n_samples,)
            Scores of all samples: Logarithms of the probabilities (or densities) of each sample in data.
            This corresponds to :meth:`sklearn.mixture.GaussianMixture.score_samples`.

            .. math::

                \left( \log \left(\sum_{i \in I} \pi_i P(x_t=i|y_t,\Phi) \right)\right)_{t \in T}

        * **score**: float
            Score: Mean of the logarithms of the probabilities (or densities)  of each sample in data.
            This corresponds to :meth:`sklearn.mixture.GaussianMixture.score`.


            .. math::

                \frac{1}{|T|} \sum_{t \in T} \left( \log \left(\sum_{i \in I} \pi_i P(x_t=i|y_t,\Phi)\right) \right)


        Parameters
        ----------
        data : np.ndarray
            The training data. Shape: (n_samples, n_components)

        Returns
        -------
        expectaions : Tuple[np.ndarray, np.ndarray, np.ndarray]

        """
        weighted_prob = self.expect(data) * self._weights[np.newaxis, :]
        # logger.debug(weighted_prob)
        with np.errstate(divide="ignore", invalid="ignore"):
            responsibilities = weighted_prob / np.sum(weighted_prob, axis=1)[:, np.newaxis]

        if np.any(np.isnan(responsibilities)):
            logger.error(
                "NaN in responsibilities (%f). Please revise your random start values",
                np.sum(np.isnan(responsibilities)),
            )
        responsibilities[np.isnan(responsibilities)] = 0
        # FIXME: if sum(weighted_prob) can contain zeros
        # (if a point is claimed by none, then the logarithm has a problem too)
        log_probabilities = np.log(np.sum(weighted_prob, axis=1))

        # logger.debug("%s %s", weighted_prob.shape, responsibilities.shape)
        assert responsibilities.shape == (
            data.shape[0],
            self.n_components,
        ), f"Wrong shape: {responsibilities.shape}"

        # logger.debug("Responsibilities: %s", responsibilities)
        return responsibilities, log_probabilities, log_probabilities.mean()

    def compute_loa(self, data: np.ndarray)-> np.ndarray:
        """
        Compute the probability that the point belongs to the model.

        Note that only works if the features return probabilities--not *densitites*.
        In case of the :class:`kuberspatiotemporal.SpatialModel`, that means you need to use the :attr:`kuberspatiotemporal.SpatialModel.box`
        attribute.

        Parameters
        ----------
        data : np.ndarray
            see :meth:`_BaseModel__expect`

        Returns
        -------
        np.ndarray
            Array with the probability of the data belonging to the model (n_samples,)
        """
        neg_probabilities = 1.0 - self.expect(data) * self._weights[np.newaxis, :]
        return 1.0 - np.prod(neg_probabilities, axis=1)

    def __maximize(self):
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
                # Trick by Heinzl2014: find the first entry >= 1, then set this and all succesive
                # elements to 1 too.
                # See this answer https://stackoverflow.com/a/16244044. Warning! the argmax solution
                # does not work, if there is no element >= 1!!

                # logger.warning(self.__priors)
                idx = np.where(self.__priors >= 1.0)[0]
                if idx.size > 0:
                    self.__priors[idx[0] :] = 1.0
                # logger.debug(self.__priors)

                # self.__priors[np.isnan(self.__priors)] = 0.0

            self._weights = self.__stick_breaking()

        self.maximize()

    def __batch(self, data: np.ndarray):
        """
        Batch learning

        Parameters
        ----------
        data : np.ndarray
            [description]
        """
        n_samples = data.shape[0]
        self.__counter = n_samples

        last_score = -np.infty
        for i in range(self.n_iterations):
            if i % 10 == 0:
                logger.info("Step %d/%d", i, self.n_iterations)

            if not data.shape[1] == self.n_dim:
                raise ValueError(f"Wrong number input dimensions: {data.shape[1]} != {self.n_dim}")

            responsibilities, temp, score = self.__expect(data)

            # Update S_0
            self._sufficient_statistics[0] = np.sum(
                responsibilities, axis=0  # (n_samples, n_components)
            )

            self.batch(data=data, responsibilities=responsibilities)

            logger.debug("Diff: %f, %f", abs(score - last_score), score)
            # Check whether to stop
            if abs(score - last_score) < 1e-3:  # self.tol:
                logger.info("Converged after %d steps", i + 1)
                # self.converged_ = True
                break
            last_score = score
            self.__maximize()
            degenerated = self.find_degenerated()
            self.__reset(degenerated)

    def __online(self, data: np.ndarray):
        """
        Online learning

        Parameters
        ----------
        data : np.ndarray
            [description]
        """

        if self.__counter == 0:
            self.__counter = 100  # FIXME Adriana! should a parameter??

            self._sufficient_statistics[0] = self._weights * 100
            self.online_init()

            logger.debug(
                "%f,%f", np.sum(self._sufficient_statistics[0]), (self.__counter) ** (-self.decay),
            )

        last_score = -np.infty
        for _ in range(self.n_iterations):
            for sample in data:

                # First reduce the influence of the older samples
                self.__counter += 1
                rate = (self.__counter) ** (-self.decay)
                for i in self._sufficient_statistics:
                    i *= 1 - rate

                # Then introduce the new sample
                responsibilities, _, score = self.__expect(sample.reshape(1, -1))

                # logger.debug('shape respons. %s', responsibilities.shape)
                self._sufficient_statistics[0] += rate * responsibilities.reshape(-1)
                logger.debug(
                    "Estimated # samples %f (%d)",
                    np.sum(self._sufficient_statistics[0]),
                    self.__counter,
                )
                # logger.debug(sample)
                self.online(
                    data=sample.reshape(1, -1), responsibilities=responsibilities, rate=rate,
                )

                self.__maximize()
                degenerated = self.find_degenerated()
                self.__reset(degenerated)

            logger.debug("Diff: %f, %f", abs(score - last_score), score)
            last_score = score

    def __reset(self, fancy_index: np.ndarray):
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

        if not self.random_reset:
            if self.nonparametric:
                self.__priors[fancy_index] = 0.0
                self._weights = self.__stick_breaking()
            else:
                self._weights[fancy_index] = 0.0
                self._weights /= np.sum(self._weights)

        self.reset(fancy_index)

    ###############################################################################################
    # Abstract methods to be implemented by specific subclases
    ###############################################################################################

    @abstractmethod
    def initialize(self):
        """Initialize the class"""

    @abstractmethod
    def batch(self, data: np.ndarray, responsibilities: np.ndarray):
        """
        Update the sufficient statistics required for batch learning

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_dim)
            Training data.
        responsibilities : np.ndarray, shape (n_samples, n_components)
            The responsibilities computed in the Expectation step
        """

    @abstractmethod
    def online_init(self):
        """Initialize the sufficient statistics required for online learning.
        Statistics should be computed from random or existing initializations"""

    @abstractmethod
    def online(
        self, data: np.ndarray, responsibilities: np.ndarray, rate: float,
    ):
        """
        Update the sufficient statistics required for online learning.

        Note that the existing statics is already multiplied with :math:`(1-\gamma)`.
        Subclasses need to *add* the update to the sufficient statistics.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_dim)
            Training data.
        responsibilities : np.ndarray, shape (n_samples, n_components)
            The responsibilities computed in the Expectation step
        rate : float,
            Adaptation rate :math:`\gamma`.
        """

    @abstractmethod
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

    @abstractmethod
    def expect(self, data: np.ndarray) -> np.ndarray:
        """
        Expectation step.

        Subclasses must compute the probability for each sample and all
        components the probability that the point belongs to
        the component.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_dim)
            Training data.

        Returns
        -------
        propabilities: np.ndarray, shape (n_samples, n_components)
        """

    @abstractmethod
    def maximize(self):
        """Optimize the model parameters based on the sufficient statistics."""

    @abstractmethod
    def find_degenerated(self) -> np.ndarray:
        """Select components that are degenerated and need to be reset"""

    @abstractmethod
    def rvs(self, n_samples: int = 1, idx: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw random value samples from the mixture distribution

        Parameters
        ----------
        n_samples: int
            How many samples to draw, by default 1

        idx : Optional[np.ndarray] (bool), shape (n_samples, n_components)
            Column selector which indicates which component has been
            selected. Only needed in case of the :class:`CompoundModel` subclass.
            by default None

        """

    ###############################################################################################
    # Methods that implement the sklearn interfaces
    ###############################################################################################

    # pylint: disable=unused-argument

    def score_samples(self, data, Y=None) -> np.ndarray:
        """See :meth:`sklearn.mixture.GaussianMixture.score_samples`"""

        if not self.loa:
            if self.score_threshold is None:
                return self.__expect(data)[1]
            else:
                print(self.__expect(data))
                print((self.__expect(data)[1] > self.score_threshold).astype(float))
                return (self.__expect(data)[1] > self.score_threshold).astype(float)
        else:
            return self.compute_loa(data)

    def score(self, data, y=None) -> float:  # pylint: disable=arguments-differ
        """See :meth:`sklearn.mixture.GaussianMixture.score`"""
        # TODO If you experience problems with score try median instead
        return self.score_samples(data).mean()

    def fit(self, data: np.ndarray, y=None):
        """See :meth:`sklearn.mixture.GaussianMixture.fit`"""

        assert data.ndim == 2, f"Data should be 2D is {data.ndim}"

        if len(data.shape) != 2:
            raise ValueError(f"Wrong input dimensions (at least 2D)")

        if data.shape[0] == 0:
            logger.info("Empty data set")
            return

        logger.info(
            "Learning %d samples (%s)", data.shape[0], "Online" if self.online_learning else "Batch"
        )

        if self.online_learning:
            self.__online(data)
        else:
            self.__batch(data)

        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """See :meth:`sklearn.mixture.GaussianMixture.predict`"""

        responsibilities, _, _ = self.__expect(data)
        return np.argmax(responsibilities, axis=1)
