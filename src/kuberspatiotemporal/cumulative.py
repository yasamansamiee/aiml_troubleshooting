# -*- coding: utf-8 -*-

r"""
Efficient computation for *boxed* cumulative density functions of Gaussian mixture models.
"""

__author__ = "Stefan Ulbrich"
__copyright__ = "Copyright 2019, Acceptto Corporation, All rights reserved."
__credits__ = ["adriana.costa@acceptto.com"]
__license__ = "Acceptto Confidential"
__version__ = "0.1"
__maintainer__ = "Stefan Ulbrich"
__email__ = "Stefan.Ulbrich@acceptto.com"
__status__ = "alpha"
__date__ = "2019-02-19"
__all__ = ['boxed_cdf']


from typing import Union
import numpy as np
from scipy.stats import mvn
from scipy.stats._multivariate import _squeeze_output
# from ..helpers.numpy import is_broadcastable

# globally disabled (because of SkLearn convention)
# pylint: disable=too-many-arguments




def boxed_cdf(
        centers: np.ndarray, width: Union[float, np.array],
        mean: np.ndarray, cov: np.ndarray,
        maxpts: float = None, abseps=1e-5, releps=1e-5) -> Union[float, np.ndarray]:
    """
    Compute the *boxed* cumulative density function given the centers and
    widths of one or more intervals.

    Parameters
    ----------
    centers : ndarray
        The centers of the boxes to compute the cumulative function for. Row matrix
        of points. Number of columns must match the dimension of the mean parameter.
    width : float or ndarray
        The width of the box. Must be able to be broadcast to the first parameter.
    mean : ndarray
        See :data:`scipy.stats.multivariate_normal`
    cov : ndarray
        See :data:`scipy.stats.multivariate_normal`
    maxpts : float
        See :data:`scipy.stats.multivariate_normal`
    abseps : float
        See :data:`scipy.stats.multivariate_normal`
    releps : float
        See :data:`scipy.stats.multivariate_normal`

    Returns
    -------
    cdf : ndarray or scalar
        Cumulative distribution function evaluated at `x`


    Notes
    -----
    The code is inspired by scipy.stats._multivariate.

    """
    # XXX SU->SU: improve doc string. Missing several parameters

    dim = mean.size

    # Increase dimensionality to 2D if necessary
    centers = np.atleast_2d(centers)
    width = np.atleast_2d(width)

    # check if dimensions are compatible
    assert centers.shape[1] == dim
    # assert is_broadcastable(centers, width)

    # We construct a matrix with the intervals defined in the rows
    # the first half of the components are the lower bound,
    # the second half the upper bound.

    lower_upper = np.hstack((centers-width/2., centers+width/2.))

    if not maxpts:
        maxpts = 1000000 * dim

    # mvnun expects 1-d arguments, so process points sequentially
    # We apply the computation along the last axis, so that we
    # process the rows in parallel.

    out = np.apply_along_axis(
        lambda stacked: mvn.mvnun(  # Computes the boxed CDF (fortran wrapper)
            stacked[0:dim],         # First columns represent the lower bound
            stacked[dim:],          # Last columns represent the upper bound
            mean, cov,              # The parameters of the normal distribution
            maxpts, abseps, releps  # Parameters of the algorithm
        )[0], -1, lower_upper
    )

    if np.any(np.isnan(out)):
        out = np.array([0])

    return _squeeze_output(out)
