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

import kuberspatiotemporal


@pytest.fixture(scope="class")
def synthetic_data():
    return None


class TestModel:

    def test_batch_em(self):
        pass

    def test_incremental_em(self):
        pass

    def test_outer_product_block_calculus(self):
        # einsum('Xi,Xj->Xij',a,a): outerproduct of each row with itself. For MxN array, leads to MxNxN.
        # np.newaxis increased the dimensionality of the tensor. numpy broadcasting inserts copies along
        # such the new axes to match the other operands dimensionality.

        data = np.random.random((100,5))

        assert np.einsum('Ti,Tj->Tij', data, data).shape == (
            data.shape[0], data.shape[1], data.shape[1]
        )  # python tuple comparison

        for i, _ in enumerate(data):
            assert np.allclose(np.einsum('Ti,Tj->Tij', data, data)[i, :, :],
                               data[i].reshape(1, -1)@data[i].reshape(-1, 1))
