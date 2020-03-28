# -*- coding: utf-8 -*-
r"""
Unit tests for the algorithm.
"""

__author__ = "Stefan Ulbrich"
__copyright__ = "Copyright 2018-2020, Acceptto Corporation, All rights reserved."
__license__ = "Acceptto Confidential"
__maintainer__ = "Stefan Ulbrich"
__email__ = "Stefan.Ulbrich@acceptto.com"
__date__ = "2020-03-26"

# pylint: disable=R,C,W

import pytest
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer


from kuberspatiotemporal import KuberModel, CompoundModel, Feature, SpatialModel

np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

N_COMPONENTS=10
N_LARGE_NUMBER=1000
N_SAMPLES=100000
N_FEATURES=4
N_SYMBOLS=5

@pytest.fixture(scope="class")
def large_data():

    ground_truth = CompoundModel(
        n_components=N_COMPONENTS,
        n_dim=N_FEATURES+2,
        nonparametric=False,
        scaling_parameter=10,
        features=
        [Feature(SpatialModel(n_dim=2, n_components=N_COMPONENTS), [0, 1])]
        +
        [
            Feature(KuberModel(n_symbols=N_SYMBOLS, nonparametric=True, n_components=N_COMPONENTS), [2+i])
            for i in range(N_FEATURES)
        ],
    )

    return ground_truth.rvs(N_SAMPLES), ground_truth


class TestLargeData:

    def test_dataset(self, large_data):
        X, ground_truth = large_data
        assert X.shape == (N_SAMPLES, N_FEATURES+2)

    def test_batch_finite_em(self, large_data, logger):


        X, ground_truth = large_data

        kst = CompoundModel(
            n_components=N_COMPONENTS,
            n_dim=N_FEATURES+2,
            n_iterations=200,
            scaling_parameter=0.5,

            nonparametric=False,
            online_learning=False,
            features=
            [Feature(SpatialModel(n_dim=2, n_components=N_COMPONENTS), [0, 1])]
            +
            [
                Feature(KuberModel(n_symbols=N_SYMBOLS, nonparametric=True, n_components=N_COMPONENTS), [ 2+i])
                for i in range(N_FEATURES)
            ],
        )

        kst.fit(X)


        idx1 = np.argsort(kst._weights)
        idx2 = np.argsort(ground_truth._weights)

        logger.debug(
            "Cmpound (C): %s, %f, %f, %f",
            kst._weights[idx1[-N_COMPONENTS*2:]],
            np.sum(kst._weights),
            kst.score(X),
            np.exp(kst.score(X)),
        )
        logger.debug(
            "Ground truth (GT): %s, %f, %f",
            ground_truth._weights[idx2[-N_COMPONENTS:]],
            ground_truth.score(X),
            np.exp(ground_truth.score(X)),
        )

        logger.debug("C: \n%s", kst.features[0].model._SpatialModel__means[idx1[-N_COMPONENTS:]])
        logger.debug("GT: \n%s", ground_truth.features[0].model._SpatialModel__means[idx2[-N_COMPONENTS:]])

        for i in range(1, N_FEATURES+1):
            logger.debug("%d)\nC:\n%s", i, kst.features[i].model._KuberModel__pmf[idx1[-N_COMPONENTS:]])
            logger.debug("GT: \n%s", ground_truth.features[i].model._KuberModel__pmf[idx2[-N_COMPONENTS:]])

        ground_truth.fit(X)
        idx2 = np.argsort(ground_truth._weights)
        logger.debug("b)\nC: \n%s", kst.features[0].model._SpatialModel__means[idx1[-N_COMPONENTS:]])
        logger.debug("GT: \n%s", ground_truth.features[0].model._SpatialModel__means[idx2[-N_COMPONENTS:]])

        for i in range(1, N_FEATURES+1):
            logger.debug("%db)\nC:\n%s", i, kst.features[i].model._KuberModel__pmf[idx1[-N_COMPONENTS:]])
            logger.debug("GT: \n%s", ground_truth.features[i].model._KuberModel__pmf[idx2[-N_COMPONENTS:]])
        logger.debug("%f, %f", ground_truth.score(X), np.exp(ground_truth.score(X)))

        assert False
        assert abs(kst.score(X) - ground_truth.score(X)) < 1e-3


    def test_batch_unparam_em(self, large_data, logger):


        X, ground_truth = large_data

        kst = CompoundModel(
            n_components=N_LARGE_NUMBER,
            n_dim=N_FEATURES+2,
            n_iterations=200,
            scaling_parameter=0.5,

            nonparametric=True,
            online_learning=False,
            features=
            # [Feature(SpatialModel(n_dim=2, box=0.01, n_components=N_LARGE_NUMBER), [0, 1])]
            [Feature(SpatialModel(n_dim=2, n_components=N_LARGE_NUMBER), [0, 1])]
            +
            [
                Feature(KuberModel(n_symbols=N_SYMBOLS, nonparametric=True, n_components=N_LARGE_NUMBER), [ 2+i])
                for i in range(N_FEATURES)
            ],
        )

        kst.fit(X)


        idx1 = np.argsort(kst._weights)
        idx2 = np.argsort(ground_truth._weights)

        logger.debug(
            "Cmpound (C): %s, %f, %f, %f",
            kst._weights[idx1[-N_COMPONENTS*2:]],
            np.sum(kst._weights),
            kst.score(X),
            np.exp(kst.score(X)),
        )

        logger.debug(
            "Ground truth (GT): %s, %f, %f",
            ground_truth._weights[idx2[-N_COMPONENTS:]],
            ground_truth.score(X),
            np.exp(ground_truth.score(X)),
        )

        logger.debug("C: \n%s", kst.features[0].model._SpatialModel__means[idx1[-N_COMPONENTS:]])
        logger.debug("GT: \n%s", ground_truth.features[0].model._SpatialModel__means[idx2[-N_COMPONENTS:]])

        for i in range(1, N_FEATURES+1):
            logger.debug("%d)\nC:\n%s", i, kst.features[i].model._KuberModel__pmf[idx1[-N_COMPONENTS:]])
            logger.debug("GT: \n%s", ground_truth.features[i].model._KuberModel__pmf[idx2[-N_COMPONENTS:]])

        # ground_truth.fit(X)
        # idx2 = np.argsort(ground_truth._weights)
        # logger.debug("b)\nC: \n%s", kst.features[0].model._SpatialModel__means[idx1[-N_COMPONENTS:]])
        # logger.debug("GT: \n%s", ground_truth.features[0].model._SpatialModel__means[idx2[-N_COMPONENTS:]])

        # for i in range(1, N_FEATURES+1):
        #     logger.debug("%db)\nC:\n%s", i, kst.features[i].model._KuberModel__pmf[idx1[-N_COMPONENTS:]])
        #     logger.debug("GT: \n%s", ground_truth.features[i].model._KuberModel__pmf[idx2[-N_COMPONENTS:]])
        # logger.debug("%f, %f", ground_truth.score(X), np.exp(ground_truth.score(X)))

        # for i in range(50):
        #     print(i, kst.features[0].model._KuberModel__pmf)
        assert False
        assert abs(kst.score(X) - ground_truth.score(X)) < 1e-3
