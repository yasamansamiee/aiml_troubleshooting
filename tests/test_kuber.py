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

from kuberspatiotemporal.kuber import KuberModel
from kuberspatiotemporal.compound import KuberspatiotemporalModel, Feature


@pytest.fixture(scope="class")
def synthetic_data():
    return None


class TestKuberModel:
    def test_batch_em(self, categorical2D):

        X, ground_truth = categorical2D
        print(X.shape)

        kst = KuberspatiotemporalModel(
            n_components=100,
            n_dim=2,
            scaling_parameter=1.1,
            nonparametric=True,
            features=[
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [0]),
                Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=100), [1]),
            ],
        )

        # print(kst._weights)

        # for i in range(50):
        #     print(i, kst.features[0].model._KuberModel__pmf)
        kst.fit(X, n_iterations=1000)
