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


from kuberspatiotemporal import CompoundModel, Feature, SpatialModel


class TestSpacialModel:
    def test_batch_finite_em(self, spatial, logger):

        X, ground_truth = spatial
        print(X.shape)

        model = SpatialModel(n_dim=2, nonparametric=False, n_iterations=200, n_components=2)


        logger.debug(" %f, %f", model.score(X), np.exp(model.score(X)))

        model.fit(X)

        idx = np.argsort(model._weights)
        logger.debug("%s, %f", model._weights[idx[-10:]], model.score(X))
        logger.debug("\n%s", model._SpatialModel__means[idx[-5:]])

        idx = np.argsort(ground_truth._weights)

        logger.debug("%s, %f", ground_truth._weights, ground_truth.score(X))
        logger.debug("\n%s", ground_truth._SpatialModel__means[idx])

        # ground_truth.fit(X)
        logger.debug("%f, %f", ground_truth.score(X), np.exp(ground_truth.score(X)))

        # for i in range(50):
        #     print(i, model.features[0].model._KuberModel__pmf)
        assert abs(model.score(X) - ground_truth.score(X)) < 1e-3
        assert False

    def test_batch_unparam_em(self, spatial, logger):

        X, ground_truth = spatial
        print(X.shape)

        model = SpatialModel(n_dim=2, nonparametric=True, scaling_parameter=1.5, n_iterations=200)


        logger.debug(" %f, %f", model.score(X), np.exp(model.score(X)))

        model.fit(X)

        idx = np.argsort(model._weights)
        logger.debug("%s, %f", model._weights[idx[-10:]], model.score(X))
        logger.debug("\n%s", model._SpatialModel__means[idx[-5:]])

        idx = np.argsort(ground_truth._weights)

        logger.debug("%s, %f", ground_truth._weights, ground_truth.score(X))
        logger.debug("\n%s", ground_truth._SpatialModel__means[idx])

        # ground_truth.fit(X)
        logger.debug("%f", ground_truth.score(X))

        # for i in range(50):
        #     print(i, model.features[0].model._KuberModel__pmf)
        assert abs(model.score(X) - ground_truth.score(X)) < 1e-3
        assert False


    def test_small_scaling_parameters(self, spatial, logger):

        X, ground_truth = spatial
        print(X.shape)

        model = SpatialModel(n_dim=2, nonparametric=True, scaling_parameter=0.15, n_iterations=200)

        logger.debug(" %f, %f", model.score(X), np.exp(model.score(X)))

        model.fit(X)

        assert not np.any(np.isnan(model._BaseModel__priors))  # pylint: disable=no-member

        idx = np.argsort(model._weights)
        logger.debug("%s, %f", model._weights[idx[-10:]], model.score(X))
        logger.debug("\n%s", model._SpatialModel__means[idx[-5:]])

        idx = np.argsort(ground_truth._weights)

        logger.debug("%s, %f", ground_truth._weights, ground_truth.score(X))
        logger.debug("\n%s", ground_truth._SpatialModel__means[idx])

    def test_incremental(self, spatial, logger):
        X, ground_truth = spatial
        print(X.shape)

        model = SpatialModel(n_dim=2, nonparametric=True, scaling_parameter=0.15, n_iterations=200, online_learning=True)
        logger.debug("Score model  %f, %f", model.score(X), np.exp(model.score(X)))

        model.fit(X)

        idx = np.argsort(model._weights)

        logger.debug(
            "Score model %s, %f, %f", model._weights[idx[-10:]], model.score(X), np.exp(model.score(X))
        )

        idx = np.argsort(model._weights)
        logger.debug("%s, %f", model._weights[idx[-10:]], model.score(X))
        logger.debug("\n%s", model._SpatialModel__means[idx[-5:]])

        idx = np.argsort(ground_truth._weights)

        logger.debug("%s, %f", ground_truth._weights, ground_truth.score(X))
        logger.debug("\n%s", ground_truth._SpatialModel__means[idx])

        assert abs(model.score(X) - ground_truth.score(X)) < 1e-2
