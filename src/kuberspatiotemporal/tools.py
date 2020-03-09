# -*- coding: utf-8 -*-
r"""
Tools to help development and testing.
"""

__author__ = "Stefan Ulbrich"
__copyright__ = "Copyright 2018-2020, Acceptto Corporation, All rights reserved."
__credits__ = "Acceptto Confidential"
__maintainer__ = "Stefan Ulbrich"
__email__ = "Stefan.Ulbrich@acceptto.com"
__date__ = "2020-03-09"


from typing import Tuple

from scipy.stats import multivariate_normal, multinomial
from sklearn.datasets import make_spd_matrix
import numpy as np
from numpy.random import randn

# Code from pyloa
def make_gmm(n_samples=100, n_features=2,
             n_clusters=3, center_std=10,
             cluster_std=1.0, weights=None) -> Tuple[np.ndarray, np.ndarray]:
    r"""Draw samples of an random Gaussian Mixture Model. Unlike make_blobs,
    covariance matrices are created randomly too.
    Parameters
    ----------
    n_samples : int or array-like, optional (default=100)
        If int, it is the total number of points equally divided among
        clusters.
        If array-like, each element of the sequence indicates
        the number of samples per cluster.
    n_features : int, optional (default=2)
        The number of features for each sample.
    n_clusters: int
        Number of clusters.
    center_std: float
        Represents the standard deviation of the
        normal distributions the centers are drawn from
    cluster_std : float
        Standard deviation of a scale factor for the covariance matrices.
    weights: array-like
        Weights of the GMM. Must have the n_samples elements.
    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """
    # pylint: disable=C0103

    if weights is None:
        weights = n_clusters*[1/n_clusters]
    else:
        assert len(weights) == n_clusters

    # Create $n$ multivariate Gaussians
    mvn = [
        multivariate_normal(
            mean=randn(n_features)*center_std,
            cov=make_spd_matrix(n_features) * np.abs(randn(1))*cluster_std
        )
        for i in range(n_clusters)
    ]

    # Draw n_samples picks from each of the n_clusters distributions.
    # We will have to discard most of them later (not too efficient)
    rvs = np.array([mvn[i].rvs(size=n_samples) for i in range(n_clusters)])

    # Damn dimensions never fit. We swap the first two axes afterwards

    # n_samples x n_clusters x n_features
    rvs = np.swapaxes(rvs, 0, 1)

    # Pick samples of the categorical distribution (i.e., multinomial with only 1 draw)

    # n_samples x n_clusters.  One `1.0` per row (all other elements `0.0`)
    idx = multinomial(1, weights, n_samples)

    # When converted to Booleans, they can be used as an element selector.
    # https://stackoverflow.com/a/23435843
    # Numpy's broadcasting takes care that this works with all values of n_features

    # pylint: disable=comparison-with-callable
    # TODO same pylint bug? Check later https://github.com/PyCQA/pylint/issues/2306
    # n_samples x n_features
    X = rvs[idx != 0]

    # From idx, we can get the class labels with a little trick and help from broadcasting
    y = np.sum(np.array([np.arange(n_clusters)]) * idx, 1)

    # Lastly, plot some statistics
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))

    return X, y