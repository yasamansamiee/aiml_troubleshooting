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

import logging
from typing import Tuple

from scipy.stats import multivariate_normal, multinomial
from sklearn.datasets import make_spd_matrix
import numpy as np
from numpy.random import randn, random_sample

try:
    import matplotlib as mpl
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Code from pyloa


def make_gmm(n_samples=100, n_features=2,
             n_clusters=3, center_std=10,
             cluster_std=.001, weights=None) -> Tuple[np.ndarray, np.ndarray]:
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
            mean=random_sample(n_features),
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
    idx = multinomial(1, weights).rvs(n_samples)

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


def make_ellipses(gmm: 'GaussianMixtureModel', ax):
    """Shamelessly stolen from
    https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html"""

    # pylint: disable=invalid-name

    if not MPL_AVAILABLE:
        raise NotImplementedError("Matplotlib not available")

    colors = ['navy', 'turquoise', 'darkorange', 'firebrick',
              'yellowgreen', 'mediumorchid', 'slateblue',
              'darkcyan', 'gold', 'mediumpurple', 'navajowhite']

    for n, color in enumerate(colors):

        if n == gmm.n_components:
            break

        covariances = gmm.covs[n]

        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means[n], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')


def check_spd(x, rtol=1e-05, atol=1e-08):
    """ Based on https://stackoverflow.com/a/42913743 and https://stackoverflow.com/a/16270026."""
    all_ev_positive = np.all(np.linalg.eigvals(x) > 0)
    is_symmetric = np.allclose(x, x.T, rtol=rtol, atol=atol)
    logger.debug("Positive definit %s", all_ev_positive)
    if not all_ev_positive:
        logger.debug("EV %s", np.linalg.eigvals(x))
    logger.debug("symmetric: %s", is_symmetric)
    return all_ev_positive and is_symmetric
