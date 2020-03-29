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

np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)




class TestKuberModel:
    def test_batch_finite_em(self, categorical2D, logger):

        X, ground_truth = categorical2D
        ground_truth.score_threshold = ground_truth.get_score_threshold(X)

        kst = CompoundModel(
            n_components=2,
            n_iterations=200,
            n_dim=2,
            scaling_parameter=1.01,
            nonparametric=False,
            score_threshold=ground_truth.score_threshold,

            features=[
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [0]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [1]),
            ],
        )

        logger.debug(" %f, %f", kst.score(X), np.exp(kst.score(X)))

        kst.fit(X)
        score = kst.score(X)


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
        assert score > 0.90

    def test_batch_unparam_em(self, categorical2D, logger):

        X, ground_truth = categorical2D
        ground_truth.score_threshold = ground_truth.get_score_threshold(X)

        kst = CompoundModel(
            n_components=100,
            n_iterations=15,
            n_dim=2,
            scaling_parameter=0.15,
            score_threshold=ground_truth.score_threshold,

            nonparametric=True,
            features=[
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [0]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [1]),
            ],
        )

        logger.debug(" %f, %f", kst.score(X), np.exp(kst.score(X)))

        kst.fit(X)
        score = kst.score(X)


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
        # assert abs(kst.score(X) - ground_truth.score(X)) < 1e-3
        assert score > 0.90

    def test_batch_pipeline_unparam_em(self, categorical2D, logger):
        def trafo(x):
            return np.array(x).reshape(-1, 1)

        X, ground_truth = categorical2D
        ground_truth.score_threshold = ground_truth.get_score_threshold(X)

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
                score_threshold=ground_truth.score_threshold,

                features=[
                    Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [0]),
                    Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [1]),
                ],
            ),
        )

        pipeline.fit(df)
        score = pipeline.score(df)

        logger.debug(pipeline.score(df))
        logger.debug("%f", ground_truth.score(X))
        logger.debug("%f", ground_truth.get_score_threshold(X))
        ground_truth.fit(X)
        logger.debug("%f", ground_truth.score(X))

        assert score > 0.90

    def test_small_scaling_parameters(self, categorical2D, logger):

        X, ground_truth = categorical2D
        ground_truth.score_threshold = ground_truth.get_score_threshold(X)

        kst = CompoundModel(
            n_components=100,
            n_iterations=20,
            n_dim=2,
            scaling_parameter=0.25,
            score_threshold=ground_truth.score_threshold,

            nonparametric=True,
            features=[
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [0]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [1]),
            ],
        )

        logger.debug(" %f, %f", kst.score(X), np.exp(kst.score(X)))

        kst.fit(X)
        score = kst.score(X)


        assert not np.any(np.isnan(kst._BaseModel__priors))  # pylint: disable=no-member

        idx = np.argsort(kst._weights)

        logger.debug("%s, %f", kst._weights[idx[-10:]], kst.score(X))

        logger.debug("\n%s", kst.features[0].model._KuberModel__pmf[idx[-5:]])
        logger.debug("\n%s", kst.features[1].model._KuberModel__pmf[idx[-5:]])

        idx = np.argsort(ground_truth._weights)
        logger.debug("%s, %f", ground_truth._weights, ground_truth.score(X))
        logger.debug("\n%s", ground_truth.features[0].model._KuberModel__pmf[idx])
        logger.debug("\n%s", ground_truth.features[1].model._KuberModel__pmf[idx])
        assert score > 0.90


    def test_incremental(self, categorical2D, logger):
        X, ground_truth = categorical2D
        ground_truth.score_threshold = ground_truth.get_score_threshold(X)


        kst = CompoundModel(
            n_components=100,
            n_iterations=2,
            n_dim=2,
            decay=0.5,
            online_learning=True,
            score_threshold=ground_truth.score_threshold,

            scaling_parameter=1,
            nonparametric=True,
            features=[
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [0]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [1]),
            ],
        )
        logger.debug("Score kst  %f, %f", kst.score(X), np.exp(kst.score(X)))

        kst.fit(X)
        score = kst.score(X)


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

        assert score > 0.90
