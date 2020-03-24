# -*- coding: utf-8 -*-
r"""
Unit tests for the algorithm.
"""

__author__ = "Stefan Ulbrich"
__copyright__ = "Copyright 2018-2020, Acceptto Corporation, All rights reserved."
__license__ = "Acceptto Confidential"
__maintainer__ = "Stefan Ulbrich"
__email__ = "Stefan.Ulbrich@acceptto.com"
__date__ = "2020-03-08"

# pylint: disable=R,C,W

import pytest
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer


from kuberspatiotemporal import KuberModel, CompoundModel, Feature, SpatialModel


@pytest.fixture(scope="class")
def synthetic_data():
    return None


class TestKuberModel:
    def test_batch_finite_em(self, heterogeneous, logger):

        X, ground_truth = heterogeneous
        print(X.shape)

        kst = CompoundModel(
            n_components=2,
            n_dim=4,
            n_iterations=200,
            scaling_parameter=1.1,
            nonparametric=False,
            features=[
                Feature(SpatialModel(n_dim=2), [0, 1]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [2]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [3]),
            ],
        )


        logger.debug(" %f, %f", kst.score(X), np.exp(kst.score(X)))

        kst.fit(X)

        idx = np.argsort(kst._weights)

        logger.debug("%s, %f", kst._weights[idx[-10:]], kst.score(X))

        logger.debug("\n%s", kst.features[0].model._KuberModel__pmf[idx[-5:]])
        logger.debug("\n%s", kst.features[1].model._KuberModel__pmf[idx[-5:]])

        idx = np.argsort(ground_truth._weights)

        logger.debug("%s, %f", ground_truth._weights, ground_truth.score(X))
        logger.debug("\n%s", ground_truth.features[0].model._KuberModel__pmf[idx])
        logger.debug("\n%s", ground_truth.features[1].model._KuberModel__pmf[idx])

        # ground_truth.fit(X)
        logger.debug("%f, %f", ground_truth.score(X), np.exp(ground_truth.score(X)))

        # for i in range(50):
        #     print(i, kst.features[0].model._KuberModel__pmf)
        assert abs(kst.score(X) - ground_truth.score(X)) < 1e-3

    def test_batch_unparam_em(self, heterogeneous, logger):

        X, ground_truth = heterogeneous
        print(X.shape)

        kst = CompoundModel(
            n_components=100,
            n_dim=4,
            n_iterations=200,
            scaling_parameter=1.1,
            nonparametric=True,
            features=[
                Feature(SpatialModel(n_dim=2), [0, 1]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [2]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [3]),
            ],
        )

        logger.debug(" %f, %f", kst.score(X), np.exp(kst.score(X)))

        kst.fit(X)

        idx = np.argsort(kst._weights)

        logger.debug("%s, %f", kst._weights[idx[-10:]], kst.score(X))

        logger.debug("\n%s", kst.features[0].model._KuberModel__pmf[idx[-5:]])
        logger.debug("\n%s", kst.features[1].model._KuberModel__pmf[idx[-5:]])

        idx = np.argsort(ground_truth._weights)

        logger.debug("%s, %f", ground_truth._weights, ground_truth.score(X))
        logger.debug("\n%s", ground_truth.features[0].model._KuberModel__pmf[idx])
        logger.debug("\n%s", ground_truth.features[1].model._KuberModel__pmf[idx])

        # ground_truth.fit(X)
        logger.debug("%f", ground_truth.score(X))

        # for i in range(50):
        #     print(i, kst.features[0].model._KuberModel__pmf)=
        assert abs(kst.score(X) - ground_truth.score(X)) < 1e-3
        assert False

    def test_small_scaling_parameters(self, heterogeneous, logger):

        X, ground_truth = heterogeneous
        print(X.shape)

        kst = CompoundModel(
            n_components=100,
            n_dim=4,
            n_iterations=200,
            scaling_parameter=0.15,
            nonparametric=True,
            features=[
                Feature(SpatialModel(n_dim=2), [0, 1]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [2]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [3]),
            ],
        )

        logger.debug(" %f, %f", kst.score(X), np.exp(kst.score(X)))

        kst.fit(X)

        assert not np.any(np.isnan(kst._BaseModel__priors))  # pylint: disable=no-member

        idx = np.argsort(kst._weights)

        logger.debug("%s, %f", kst._weights[idx[-10:]], kst.score(X))

        logger.debug("\n%s", kst.features[0].model._KuberModel__pmf[idx[-5:]])
        logger.debug("\n%s", kst.features[1].model._KuberModel__pmf[idx[-5:]])

        idx = np.argsort(ground_truth._weights)
        logger.debug("%s, %f", ground_truth._weights, ground_truth.score(X))
        logger.debug("\n%s", ground_truth.features[0].model._KuberModel__pmf[idx])
        logger.debug("\n%s", ground_truth.features[1].model._KuberModel__pmf[idx])

    def test_incremental(self, heterogeneous, logger):
        X, ground_truth = heterogeneous
        print(X.shape)

        kst = CompoundModel(
            n_components=100,
            n_dim=4,
            n_iterations=200,
            scaling_parameter=0.15,
            alpha=0.5,
            online_learning=True,
            nonparametric=True,
            features=[
                Feature(SpatialModel(n_dim=2), [0, 1]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [2]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [3]),
            ],
        )
        logger.debug("Score kst  %f, %f", kst.score(X), np.exp(kst.score(X)))

        kst.fit(X)

        idx = np.argsort(kst._weights)

        logger.debug(
            "Score kst %s, %f, %f", kst._weights[idx[-10:]], kst.score(X), np.exp(kst.score(X))
        )

        logger.debug("\n%s", kst.features[0].model._KuberModel__pmf[idx[-5:]])
        logger.debug("\n%s", kst.features[1].model._KuberModel__pmf[idx[-5:]])

        idx = np.argsort(ground_truth._weights)
        logger.debug(
            "Score ground truth %s, %f, %f",
            ground_truth._weights,
            ground_truth.score(X),
            np.exp(ground_truth.score(X)),
        )
        logger.debug("\n%s", ground_truth.features[0].model._KuberModel__pmf[idx])
        logger.debug("\n%s", ground_truth.features[1].model._KuberModel__pmf[idx])

        assert abs(kst.score(X) - ground_truth.score(X)) < 1e-2
