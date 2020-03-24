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


from kuberspatiotemporal import KuberModel, CompoundModel, Feature


@pytest.fixture(scope="class")
def synthetic_data():
    return None


class TestKuberModel:
    def test_batch_finite_em(self, categorical2D, logger):

        X, ground_truth = categorical2D
        print(X.shape)

        kst = CompoundModel(
            n_components=2,
            n_iterations=200,
            n_dim=2,
            scaling_parameter=1.01,
            nonparametric=False,
            features=[
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [0]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [1]),
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

    def test_batch_unparam_em(self, categorical2D, logger):

        X, ground_truth = categorical2D
        print(X.shape)

        kst = CompoundModel(
            n_components=100,
            n_iterations=15,
            n_dim=2,
            scaling_parameter=0.15,
            nonparametric=True,
            features=[
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [0]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [1]),
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

    def test_batch_pipeline_unparam_em(self, categorical2D, logger):
        def trafo(x):
            return np.array(x).reshape(-1, 1)

        X, ground_truth = categorical2D
        print(X.shape)

        df = pd.DataFrame(X, columns=["f1", "f2"])

        pipeline = make_pipeline(
            make_column_transformer(
                (FunctionTransformer(trafo), "f1"), (FunctionTransformer(trafo), "f2")
            ),
            CompoundModel(
                n_components=100,
                n_iterations=15,
                n_dim=2,
                scaling_parameter=1.11,
                nonparametric=True,
                features=[
                    Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [0]),
                    Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [1]),
                ],
            ),
        )

        pipeline.fit(df)
        logger.debug(pipeline.score(df))
        logger.debug("%f", ground_truth.score(X))
        ground_truth.fit(X)
        logger.debug("%f", ground_truth.score(X))

        assert abs(pipeline.score(df) - ground_truth.score(X)) < 1e-3

    def test_small_scaling_parameters(self, categorical2D, logger):

        X, ground_truth = categorical2D
        print(X.shape)

        kst = CompoundModel(
            n_components=100,
            n_iterations=20,
            n_dim=2,
            scaling_parameter=0.25,
            nonparametric=True,
            features=[
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [0]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [1]),
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

    def test_incremental(self, categorical2D, logger):
        X, ground_truth = categorical2D
        print(X.shape)

        kst = CompoundModel(
            n_components=100,
            n_iterations=2,
            n_dim=2,
            alpha=0.5,
            online_learning=True,
            scaling_parameter=1,
            nonparametric=True,
            features=[
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [0]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [1]),
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
