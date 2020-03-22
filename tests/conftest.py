# pylint: disable=R,C,W


import logging
import pytest
from scipy.stats import multinomial
from numpy.random import dirichlet
import numpy as np



from kuberspatiotemporal.kuber import KuberModel
from kuberspatiotemporal.compound import KuberspatiotemporalModel, Feature
logging.basicConfig(format='[%(levelname)s:%(name)s:%(funcName)s:%(lineno)d] %(message)s',level=logging.DEBUG)

@pytest.fixture(scope="module")
def categorical2D():
    def categorical_mixture(n_draws: int):
        pi = dirichlet([1, 1], 1)[0]
        pm11, pm12, pm21, pm22 = dirichlet([1, 1, 1], 4)
        print("pi: ", pi)
        print("Component1: ", pm11, pm21)
        print("Component2: ", pm12, pm22)
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
        return np.array([feat1[idx != 0], feat2[idx != 0]]).T, pi, pm11, pm12, pm21, pm22

    X, pi, pm11, pm12, pm21, pm22 = categorical_mixture(1000)

    ground_truth = KuberspatiotemporalModel(
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
    ground_truth.features[0].model._KuberModel__pmf = np.array([pm11,pm12])
    ground_truth.features[1].model._KuberModel__pmf = np.array([pm21,pm22])
    ground_truth._weights = pi

    return X, ground_truth


@pytest.fixture
def logger():
    logger_ = logging.getLogger('UnitTest')
    return logger_