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

np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)


class TestKuberspatialModel:
    def test_batch_finite_em(self, heterogeneous, logger):

        # Checks wether 90+% of samples are classified
        # as part of the model.

        # Data is comparatively simple but uses
        # sklearn's pipelining tools

        X, gt_pipeline, ground_truth = heterogeneous

        kst = CompoundModel(
            n_components=2,
            n_dim=4,
            n_iterations=200,
            scaling_parameter=1.1,
            nonparametric=False,
            online_learning=False,
            score_threshold=ground_truth.score_threshold,  # has been set in the fixture
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

        logger.debug(ground_truth.score_threshold)

        score = pipeline.score(X)
        gt_score = gt_pipeline.score(X)


        idx = np.argsort(kst._weights)

        logger.debug("Cmpound (C): %s, %f", kst._weights[idx[-10:]], score)

        logger.debug("\n%s", kst.features[0].model._SpatialModel__means[idx[-5:]])
        logger.debug("\n%s", kst.features[1].model._KuberModel__pmf[idx[-5:]])
        logger.debug("\n%s", kst.features[2].model._KuberModel__pmf[idx[-5:]])

        idx = np.argsort(ground_truth._weights)

        logger.debug(
            "Ground truth (GT): %s, %f", ground_truth._weights[idx], gt_score,
        )
        logger.debug("\n%s", ground_truth.features[0].model._SpatialModel__means[idx[-5:]])
        logger.debug("\n%s", ground_truth.features[1].model._KuberModel__pmf[idx])
        logger.debug("\n%s", ground_truth.features[2].model._KuberModel__pmf[idx])

        assert score > 0.90

    def test_batch_unparam_em(self, heterogeneous, logger):

        # This test fails usually,
        # showing the importance of a smaller (in this case)
        # or, more generally, carefully chosen scaling parameter

        X, gt_pipeline, ground_truth = heterogeneous

        kst = CompoundModel(
            n_components=20,
            n_dim=4,
            n_iterations=200,
            scaling_parameter=10,
            nonparametric=True,
            online_learning=False,
            score_threshold=ground_truth.score_threshold,  # has been set in the fixture
            features=[
                Feature(SpatialModel(n_dim=2, n_components=20, box=0.1), [0, 1]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=20), [2]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=20), [3]),
            ],
        )

        Y= kst.rvs(500)

        pipeline = make_pipeline(
            make_column_transformer(
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "x"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "y"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "f1"),
                (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "f2"),
            ),
            kst,
        )
        pipeline.fit(X[:500])
        kst.score_threshold = kst.get_score_threshold(X[:500].to_numpy())

        score = pipeline.score(X[500:])

        assert kst.score(Y) < 0.4

        gt_score = gt_pipeline.score(X)
        logger.debug(gt_pipeline.score(X))

        idx = np.argsort(kst._weights)

        logger.debug(ground_truth.score_threshold)
        logger.debug("Cmpound (C): %s, %f", kst._weights[idx[-10:]], score)

        logger.debug("\n%s", kst.features[0].model._SpatialModel__means[idx[-5:]])
        logger.debug("\n%s", kst.features[1].model._KuberModel__pmf[idx[-5:]])
        logger.debug("\n%s", kst.features[2].model._KuberModel__pmf[idx[-5:]])

        idx = np.argsort(ground_truth._weights)

        logger.debug(
            "Ground truth (GT): %s, %f", ground_truth._weights[idx], gt_score,
        )
        logger.debug("\n%s", ground_truth.features[0].model._SpatialModel__means[idx[-5:]])
        logger.debug("\n%s", ground_truth.features[1].model._KuberModel__pmf[idx])
        logger.debug("\n%s", ground_truth.features[2].model._KuberModel__pmf[idx])

        gt_pipeline.fit(X)

        assert score > 0.90

    def test_small_scaling_parameters(self, heterogeneous, logger):

        X, gt_pipeline, ground_truth = heterogeneous

        kst = CompoundModel(
            n_components=100,
            n_dim=4,
            n_iterations=200,
            scaling_parameter=0.15,
            nonparametric=True,
            online_learning=False,
            score_threshold=ground_truth.score_threshold,  # has been set in the fixture

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

        score = pipeline.score(X)
        gt_score = gt_pipeline.score(X)
        logger.debug(gt_pipeline.score(X))

        idx = np.argsort(kst._weights)

        logger.debug(ground_truth.score_threshold)
        logger.debug("Cmpound (C): %s, %f", kst._weights[idx[-10:]], score)

        logger.debug("\n%s", kst.features[0].model._SpatialModel__means[idx[-5:]])
        logger.debug("\n%s", kst.features[1].model._KuberModel__pmf[idx[-5:]])
        logger.debug("\n%s", kst.features[2].model._KuberModel__pmf[idx[-5:]])

        idx = np.argsort(ground_truth._weights)

        logger.debug(
            "Ground truth (GT): %s, %f", ground_truth._weights[idx], gt_score,
        )
        logger.debug("\n%s", ground_truth.features[0].model._SpatialModel__means[idx[-5:]])
        logger.debug("\n%s", ground_truth.features[1].model._KuberModel__pmf[idx])
        logger.debug("\n%s", ground_truth.features[2].model._KuberModel__pmf[idx])

        gt_pipeline.fit(X)

        assert score > 0.90


    def test_incremental(self, heterogeneous, logger):
        X, gt_pipeline, ground_truth = heterogeneous

        kst = CompoundModel(
            n_components=100,
            n_dim=4,
            decay=0.5,
            n_iterations=1,
            scaling_parameter=0.5,
            nonparametric=True,
            online_learning=False,
            score_threshold=ground_truth.score_threshold,
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

        score = pipeline.score(X)
        gt_score = gt_pipeline.score(X)
        logger.debug(gt_pipeline.score(X))

        idx = np.argsort(kst._weights)

        logger.debug(ground_truth.score_threshold)
        logger.debug("Cmpound (C): %s, %f", kst._weights[idx[-10:]], score)

        logger.debug("\n%s", kst.features[0].model._SpatialModel__means[idx[-5:]])
        logger.debug("\n%s", kst.features[1].model._KuberModel__pmf[idx[-5:]])
        logger.debug("\n%s", kst.features[2].model._KuberModel__pmf[idx[-5:]])

        idx = np.argsort(ground_truth._weights)

        logger.debug(
            "Ground truth (GT): %s, %f", ground_truth._weights[idx], gt_score,
        )
        logger.debug("\n%s", ground_truth.features[0].model._SpatialModel__means[idx[-5:]])
        logger.debug("\n%s", ground_truth.features[1].model._KuberModel__pmf[idx])
        logger.debug("\n%s", ground_truth.features[2].model._KuberModel__pmf[idx])

        gt_pipeline.fit(X)

        assert score > 0.90