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


class TestKuberspatialModel:
    def test_batch_finite_em(self, heterogeneous, logger):

        X, gt_pipeline, ground_truth = heterogeneous

        kst = CompoundModel(
            n_components=2,
            n_dim=4,
            n_iterations=200,
            scaling_parameter=1.1,
            nonparametric=False,
            online_learning=False,
            features=[
                Feature(SpatialModel(n_dim=2, n_components=2), [0, 1]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [2]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [3]),
            ],
        )

        pipeline = make_pipeline(
            make_column_transformer(
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "x"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "y"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "f1"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "f2"),
            ),
            kst,
        )


        pipeline.fit(X)

        idx = np.argsort(kst._weights)

        logger.debug("%s, %f", kst._weights[idx[-10:]], pipeline.score(X), np.exp(pipeline.score(X)))

        logger.debug("\n%s", kst.features[0].model._SpatialModel__means[idx[-5:]])
        logger.debug("\n%s", kst.features[1].model._KuberModel__pmf[idx[-5:]])
        logger.debug("\n%s", kst.features[2].model._KuberModel__pmf[idx[-5:]])

        idx = np.argsort(ground_truth._weights)

        logger.debug("%s, %f", ground_truth._weights, gt_pipeline.score(X), np.exp(gt_pipeline.score(X)))
        logger.debug("\n%s", ground_truth.features[0].model._SpatialModel__means[idx[-5:]])
        logger.debug("\n%s", ground_truth.features[1].model._KuberModel__pmf[idx])
        logger.debug("\n%s", ground_truth.features[2].model._KuberModel__pmf[idx])

        # ground_truth.fit(X)
        logger.debug("%f, %f", gt_pipeline.score(X), np.exp(gt_pipeline.score(X)))

        # for i in range(50):
        #     print(i, kst.features[0].model._KuberModel__pmf)
        assert abs(pipeline.score(X) - gt_pipeline.score(X)) < 1e-3

    def test_batch_unparam_em(self, heterogeneous, logger):


        X, gt_pipeline, ground_truth = heterogeneous

        kst = CompoundModel(
            n_components=100,
            n_dim=4,
            n_iterations=200,
            scaling_parameter=10,
            nonparametric=True,
            online_learning=False,
            features=[
                Feature(SpatialModel(n_dim=2, n_components=100, box=0.1), [0, 1]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [2]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [3]),
            ],
        )

        pipeline = make_pipeline(
            make_column_transformer(
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "x"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "y"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "f1"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "f2"),
            ),
            kst,
        )


        pipeline.fit(X)


        # kst.features[0].model.box = .4

        idx = np.argsort(kst._weights)

        logger.debug("%s, %f, %f, %f", kst._weights[idx[-10:]], np.sum(kst._weights),  pipeline.score(X), np.exp(pipeline.score(X)))

        logger.debug("\n%s", kst.features[0].model._SpatialModel__means[idx[-5:]])
        logger.debug("\n%s", kst.features[1].model._KuberModel__pmf[idx[-5:]])
        logger.debug("\n%s", kst.features[2].model._KuberModel__pmf[idx[-5:]])

        idx = np.argsort(ground_truth._weights)
        ground_truth.features[0].model.box = .1
        logger.debug("%s, %f, %f", ground_truth._weights, gt_pipeline.score(X), np.exp(gt_pipeline.score(X)))
        logger.debug("\n%s", ground_truth.features[0].model._SpatialModel__means[idx[-5:]])
        logger.debug("\n%s", ground_truth.features[1].model._KuberModel__pmf[idx])
        logger.debug("\n%s", ground_truth.features[2].model._KuberModel__pmf[idx])

        # ground_truth.fit(X)
        logger.debug("%f, %f", gt_pipeline.score(X), np.exp(gt_pipeline.score(X)))

        # for i in range(50):
        #     print(i, kst.features[0].model._KuberModel__pmf)
        assert abs(pipeline.score(X) - gt_pipeline.score(X)) < 1e-3

    def test_small_scaling_parameters(self, heterogeneous, logger):

        X, gt_pipeline, ground_truth = heterogeneous

        kst = CompoundModel(
            n_components=100,
            n_dim=4,
            n_iterations=200,
            scaling_parameter=0.15,
            nonparametric=True,
            online_learning=False,
            features=[
                Feature(SpatialModel(n_dim=2, n_components=100), [0, 1]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [2]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [3]),
            ],
        )

        pipeline = make_pipeline(
            make_column_transformer(
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "x"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "y"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "f1"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "f2"),
            ),
            kst,
        )


        pipeline.fit(X)

        idx = np.argsort(kst._weights)

        logger.debug("%s, %f", kst._weights[idx[-10:]], pipeline.score(X))

        logger.debug("\n%s", kst.features[0].model._SpatialModel__means[idx[-5:]])
        logger.debug("\n%s", kst.features[1].model._KuberModel__pmf[idx[-5:]])
        logger.debug("\n%s", kst.features[2].model._KuberModel__pmf[idx[-5:]])

        idx = np.argsort(ground_truth._weights)

        logger.debug("%s, %f", ground_truth._weights, gt_pipeline.score(X))
        logger.debug("\n%s", ground_truth.features[0].model._SpatialModel__means[idx[-5:]])
        logger.debug("\n%s", ground_truth.features[1].model._KuberModel__pmf[idx])
        logger.debug("\n%s", ground_truth.features[2].model._KuberModel__pmf[idx])

        # ground_truth.fit(X)
        logger.debug("%f, %f", gt_pipeline.score(X), np.exp(gt_pipeline.score(X)))

        # for i in range(50):
        #     print(i, kst.features[0].model._KuberModel__pmf)
        assert abs(pipeline.score(X) - gt_pipeline.score(X)) < 1e-3

    def test_incremental(self, heterogeneous, logger):
        X, gt_pipeline, ground_truth = heterogeneous

        kst = CompoundModel(
            n_components=100,
            n_dim=4,
            alpha=0.5,
            n_iterations=1,
            scaling_parameter=0.5,
            nonparametric=True,
            online_learning=False,
            features=[
                Feature(SpatialModel(n_dim=2, n_components=100), [0, 1]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [2]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [3]),
            ],
        )

        pipeline = make_pipeline(
            make_column_transformer(
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "x"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "y"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "f1"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "f2"),
            ),
            kst,
        )


        pipeline.fit(X)

        idx = np.argsort(kst._weights)

        logger.debug("%s, %f", kst._weights[idx[-10:]], pipeline.score(X))

        logger.debug("\n%s", kst.features[0].model._SpatialModel__means[idx[-5:]])
        logger.debug("\n%s", kst.features[1].model._KuberModel__pmf[idx[-5:]])
        logger.debug("\n%s", kst.features[2].model._KuberModel__pmf[idx[-5:]])

        idx = np.argsort(ground_truth._weights)

        logger.debug("%s, %f", ground_truth._weights, gt_pipeline.score(X))
        logger.debug("\n%s", ground_truth.features[0].model._SpatialModel__means[idx[-5:]])
        logger.debug("\n%s", ground_truth.features[1].model._KuberModel__pmf[idx])
        logger.debug("\n%s", ground_truth.features[2].model._KuberModel__pmf[idx])

        # ground_truth.fit(X)
        logger.debug("%f, %f", gt_pipeline.score(X), np.exp(gt_pipeline.score(X)))

        # for i in range(50):
        #     print(i, kst.features[0].model._KuberModel__pmf)
        assert abs(pipeline.score(X) - gt_pipeline.score(X)) < 1e-3


