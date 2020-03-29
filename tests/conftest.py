# pylint: disable=R,C,W


import logging
import pytest
from scipy.stats import multinomial, multivariate_normal
from numpy.random import dirichlet
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.datasets import make_spd_matrix
from sklearn.pipeline import make_pipeline, Pipeline



from kuberspatiotemporal import KuberModel, CompoundModel, Feature, SpatialModel

logging.basicConfig(
    format="[%(levelname)s:%(name)s:%(funcName)s:%(lineno)d] %(message)s", level=logging.DEBUG
)


def categorical_mixture(n_draws: int):
    pi = dirichlet([1, 1], 1)[0]
    pm11, pm12, pm21, pm22 = dirichlet([1, 1, 1], 4)
    # print("pi: ", pi)
    # print("Component1: ", pm11, pm21)
    # print("Component2: ", pm12, pm22)
    idx = multinomial(1, pi).rvs(size=n_draws)
    feat1 = np.array(
        [
            [np.where(r == 1)[0][0] for r in mn.rvs(size=n_draws)]
            for mn in [multinomial(1, pm11), multinomial(1, pm12)]
        ]
    ).T
    feat2 = np.array(
        [
            [np.where(r == 1)[0][0] for r in mn.rvs(size=n_draws)]
            for mn in [multinomial(1, pm21), multinomial(1, pm22)]
        ]
    ).T
    # print(idx,feat)
    # pylint: disable=unsubscriptable-object
    return np.array([feat1[idx != 0], feat2[idx != 0]]).T, pi, pm11, pm12, pm21, pm22, idx


def heterogeneous_mixture(n_draws: int):
    X1, pi, pm11, pm12, pm21, pm22, idx = categorical_mixture(n_draws)

    n_clusters = 2
    cluster_std = 0.005
    n_features = 2

    means = np.random.random((n_clusters, n_features))
    covs = [
        make_spd_matrix(n_features) * np.abs(np.random.randn(1)) * cluster_std
        for i in range(n_clusters)
    ]
    mvn = [multivariate_normal(mean=means[i], cov=covs[i],) for i in range(n_clusters)]
    rvs_x = np.array([mvn[i].rvs(size=n_draws) for i in range(n_clusters)])
    rvs_x = np.swapaxes(rvs_x, 0, 1)
    X2 = rvs_x[idx != 0]

    return np.hstack((X2, X1)), pi, means, np.array(covs), pm11, pm12, pm21, pm22


@pytest.fixture(scope="module")
def heterogeneous():
    X, pi, means, covs, pm11, pm12, pm21, pm22 = heterogeneous_mixture(1000)

    df = pd.DataFrame(X, columns=["x", "y", "f1", "f2"])

    ct = make_column_transformer(
        (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "x"),
        (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "y"),
        (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "f1"),
        (FunctionTransformer(lambda x: np.array(x).reshape(-1, 1)), "f2"),
    )
    ct.fit(df)
    ground_truth = CompoundModel(
        n_components=2,
        n_dim=4,
        scaling_parameter=1.1,
        nonparametric=True,
        features=[
            Feature(SpatialModel(n_dim=2,  n_components=2),[0,1]),
            Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [2]),
            Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [3]),
        ],
    )

    ground_truth.features[0].model._SpatialModel__means = means
    ground_truth.features[0].model._SpatialModel__covs = covs

    ground_truth.features[1].model._KuberModel__pmf = np.array([pm11, pm12])
    ground_truth.features[2].model._KuberModel__pmf = np.array([pm21, pm22])
    ground_truth._weights = pi

    ground_truth.score_threshold = ground_truth.get_score_threshold(X, 0.005)



    return df, make_pipeline(ct,ground_truth), ground_truth


@pytest.fixture(scope="module")
def spatial():
    X, pi, means, covs, pm11, pm12, pm21, pm22 = heterogeneous_mixture(1000)
    ground_truth = SpatialModel(n_dim=2, n_components=2)
    ground_truth._SpatialModel__means = means
    ground_truth._SpatialModel__covs = covs
    return X[:,:2], ground_truth


@pytest.fixture(scope="module")
def categorical2D():
    X, pi, pm11, pm12, pm21, pm22 = categorical_mixture(1000)

    ground_truth = CompoundModel(
        n_components=2,
        n_dim=2,
        scaling_parameter=1.1,
        nonparametric=True,
        features=[
            Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [0]),
            Feature(KuberModel(n_symbols=3, nonparametric=True, n_components=2), [1]),
        ],
    )
    # pylint: disable=protected-access
    ground_truth.features[0].model._KuberModel__pmf = np.array([pm11, pm12])
    ground_truth.features[1].model._KuberModel__pmf = np.array([pm21, pm22])
    ground_truth._weights = pi

    return X, ground_truth


@pytest.fixture
def logger():
    logger_ = logging.getLogger("UnitTest")
    return logger_
