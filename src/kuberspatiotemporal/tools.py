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
import sys
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
             cluster_std=.005, n_symbols=5, weights=None) -> Tuple[np.ndarray, np.ndarray]:

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
    
    P = np.random.dirichlet([1]*n_symbols,n_clusters)
    mn = [
        multinomial(n=1,p=p)
        for p in P
    ]

    # Draw n_samples picks from each of the n_clusters distributions.
    # We will have to discard most of them later (not too efficient)
    rvs_x = np.array([mvn[i].rvs(size=n_samples) for i in range(n_clusters)])
    
    # https://stackoverflow.com/a/42497456
    rvs_s = np.array( [[np.where(r==1)[0][0] for r in i.rvs(size=n_samples)] for i in mn ])
    
    # Damn dimensions never fit. We swap the first two axes afterwards

    # n_samples x n_clusters x n_features
    rvs_x = np.swapaxes(rvs_x, 0, 1)
    rvs_s = rvs_s.T
    


    # Pick samples of the categorical distribution (i.e., multinomial with only 1 draw)

    # n_samples x n_clusters.  One `1.0` per row (all other elements `0.0`)
    idx = multinomial(1, weights).rvs(n_samples)

    # When converted to Booleans, they can be used as an element selector.
    # https://stackoverflow.com/a/23435843
    # Numpy's broadcasting takes care that this works with all values of n_features

    # pylint: disable=comparison-with-callable
    # TODO same pylint bug? Check later https://github.com/PyCQA/pylint/issues/2306
    # n_samples x n_features
    X = rvs_x[idx != 0]
    
    S = rvs_s[idx != 0] # pylint: disable=unsubscriptable-object

    # From idx, we can get the class labels with a little trick and help from broadcasting
    y = np.sum(np.array([np.arange(n_clusters)]) * idx, 1)

    # Lastly, plot some statistics
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))

    return X,S, y, P

def check_singular(x, tol=1/sys.float_info.epsilon):
    is_singular = np.linalg.cond(x) > tol
    if is_singular:
        logger.warning("Found singular matrix:\n%s\ncond: %f>%f",
                       x, np.linalg.cond(x), tol)
    return is_singular

def check_spd(x, rtol=1e-05, atol=1e-08):
    """ Based on https://stackoverflow.com/a/42913743 and https://stackoverflow.com/a/16270026."""
    all_ev_positive = np.all(np.linalg.eigvals(x) > 0)
    is_symmetric = np.allclose(x, x.T, rtol=rtol, atol=atol)
    if not all_ev_positive or not is_symmetric:
        logger.warning("Matrix not SPD:\n%s", x)
        logger.warning("Positive definit %s", all_ev_positive)
        logger.warning("symmetric: %s", is_symmetric)
        logger.warning("EV %s", np.linalg.eigvals(x))
    return all_ev_positive and is_symmetric


def make_ellipses(gmm: 'GaussianMixtureModel', ax, min_weight=.0):
    """Shamelessly stolen from
    https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html"""

    # pylint: disable=invalid-name

    if not MPL_AVAILABLE:
        raise NotImplementedError("Matplotlib not available")

    colors = ['navy', 'turquoise', 'darkorange', 'firebrick',
              'yellowgreen', 'mediumorchid', 'slateblue',
              'darkcyan', 'gold', 'mediumpurple', 'navajowhite']*30

    for n, color in enumerate(colors):

        if n == gmm.n_components:
            break

        if gmm.weights[n] < min_weight:
            continue
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

def repr_list_ndarray(x: np.ndarray) -> str:
    return f"`list of length {len(x)}, elements of shape {np.asarray(x).shape[1:]}`"

def repr_ndarray(x: np.ndarray) -> str:
    return f"Array {x.shape}" if x is not None else "None"


def cholesky_precisions(x: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    n_components, n_features, _ = x.shape
    
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(x):
        try:
            cov_chol = scipy_linalg.cholesky(covariance, lower=True)
            # cov_chol2 = np.linalg.cholesky(covariance)
            # assert np.allclose(cov_chol,cov_chol2)
        except scipy_linalg.LinAlgError:
            raise ValueError("Could not invert")
        precisions_chol[k] = scipy_linalg.solve_triangular(cov_chol,
                                                     np.eye(n_features),
                                                     lower=True).T # cannot be vectorized :(
        
        
    
    n_components, _, _ = precisions_chol.shape
    log_det_chol = (np.sum(np.log(
        precisions_chol.reshape(
            n_components, -1)[:, ::n_features + 1]), 1))
    return precisions_chol, log_det_chol