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

N_COMPONENTS = 10
N_LARGE_NUMBER = 100
N_SAMPLES = 100000
N_FEATURES = 5
N_SYMBOLS = 5
# N_COMPONENTS=10
# N_LARGE_NUMBER=100
# N_SAMPLES=1000000
# N_FEATURES=10
# N_SYMBOLS=20


@pytest.fixture(scope="class")
def large_data():

    ground_truth = CompoundModel(
        n_components=N_COMPONENTS,
        n_dim=N_FEATURES + 2,
        nonparametric=False,
        scaling_parameter=10,
        features=[Feature(SpatialModel(n_dim=2, n_components=N_COMPONENTS), [0, 1])]
        + [
            Feature(
                KuberModel(n_symbols=N_SYMBOLS, nonparametric=True, n_components=N_COMPONENTS),
                [2 + i],
            )
            for i in range(N_FEATURES)
        ],
    )

    return ground_truth.rvs(N_SAMPLES), ground_truth


class TestLargeData:
    def test_dataset(self, large_data):
        X, ground_truth = large_data
        assert X.shape == (N_SAMPLES, N_FEATURES + 2)

    def test_batch_finite_em(self, large_data, logger):

        # Test learning with prior knowledge about the number of components
        # First check that the acceptance rate over the test data
        # is below 50% (pick the threshold from the ground truth)
        # Then learn and expect the prediction rate to be >90%

        X, ground_truth = large_data
        ground_truth.score_threshold = ground_truth.get_score_threshold(X)

        kst = CompoundModel(
            n_components=N_COMPONENTS,
            n_dim=N_FEATURES + 2,
            n_iterations=200,
            scaling_parameter=0.5,
            nonparametric=False,
            online_learning=False,
            score_threshold=ground_truth.score_threshold,

            features=[Feature(SpatialModel(n_dim=2, n_components=N_COMPONENTS), [0, 1])]
            + [
                Feature(
                    KuberModel(n_symbols=N_SYMBOLS, nonparametric=True, n_components=N_COMPONENTS),
                    [2 + i],
                )
                for i in range(N_FEATURES)
            ],
        )
        score = kst.score(X)

        logger.debug("Compound (C): Score before learning: %f", score)
        assert score < 0.50

        kst.fit(X)

        idx1 = np.argsort(kst._weights)
        idx2 = np.argsort(ground_truth._weights)

        score = kst.score(X)

        logger.debug(
            "Cmpound (C): %s, %f",
            kst._weights[idx1[-N_COMPONENTS * 2 :]],
            score
            # np.exp(kst.score(X)),
        )

        gt_score = ground_truth.score(X)
        logger.debug(
            "Ground truth (GT): %s, %f",
            ground_truth._weights[idx2[-N_COMPONENTS:]],
            gt_score,
            # np.exp(ground_truth.score(X)),
        )

        logger.debug("C: \n%s", kst.features[0].model._SpatialModel__means[idx1[-N_COMPONENTS:]])
        logger.debug(
            "GT: \n%s", ground_truth.features[0].model._SpatialModel__means[idx2[-N_COMPONENTS:]]
        )

        for i in range(1, N_FEATURES + 1):
            logger.debug(
                "%d)\nC:\n%s", i, kst.features[i].model._KuberModel__pmf[idx1[-N_COMPONENTS:]]
            )
            logger.debug(
                "GT: \n%s", ground_truth.features[i].model._KuberModel__pmf[idx2[-N_COMPONENTS:]]
            )


        assert score > 0.90

    def test_batch_unparam_em(self, large_data, logger):


        # Same as the test above but without knowledge about the number of clusters

        X, ground_truth = large_data

        # np.savetxt('data.csv.gz',X,delimiter=',')

        ground_truth.score_threshold = ground_truth.get_score_threshold(X)

        logger.debug("Threshold: %f", ground_truth.score_threshold)

        kst = CompoundModel(
            n_components=N_LARGE_NUMBER,
            n_dim=N_FEATURES + 2,
            n_iterations=200,
            scaling_parameter=0.5,
            nonparametric=True,
            online_learning=False,
            score_threshold=ground_truth.score_threshold,
            features=
            # [Feature(SpatialModel(n_dim=2, box=0.01, n_components=N_LARGE_NUMBER), [0, 1])]
            [Feature(SpatialModel(n_dim=2, n_components=N_LARGE_NUMBER), [0, 1])]
            + [
                Feature(
                    KuberModel(
                        n_symbols=N_SYMBOLS, nonparametric=True, n_components=N_LARGE_NUMBER
                    ),
                    [2 + i],
                )
                for i in range(N_FEATURES)
            ],
        )

        score = kst.score(X)

        logger.debug("Compound (C): Score before learning: %f", score)
        assert score < 0.50

        kst.fit(X)

        idx1 = np.argsort(kst._weights)
        idx2 = np.argsort(ground_truth._weights)

        score = kst.score(X)

        logger.debug(
            "Cmpound (C): %s, %f",
            kst._weights[idx1[-N_COMPONENTS * 2 :]],
            score
            # np.exp(kst.score(X)),
        )

        gt_score = ground_truth.score(X)
        logger.debug(
            "Ground truth (GT): %s, %f",
            ground_truth._weights[idx2[-N_COMPONENTS:]],
            gt_score,
            # np.exp(ground_truth.score(X)),
        )

        logger.debug("C: \n%s", kst.features[0].model._SpatialModel__means[idx1[-N_COMPONENTS:]])
        logger.debug(
            "GT: \n%s", ground_truth.features[0].model._SpatialModel__means[idx2[-N_COMPONENTS:]]
        )

        for i in range(1, N_FEATURES + 1):
            logger.debug(
                "%d)\nC:\n%s", i, kst.features[i].model._KuberModel__pmf[idx1[-N_COMPONENTS:]]
            )
            logger.debug(
                "GT: \n%s", ground_truth.features[i].model._KuberModel__pmf[idx2[-N_COMPONENTS:]]
            )


        assert score > 0.90

        assert False

