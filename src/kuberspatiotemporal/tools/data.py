# -*- coding: utf-8 -*-
r"""
Tools to help development and testing.
"""

__author__ = "Adriana Costa"
__copyright__ = "Copyright 2018-2020, Acceptto Corporation, All rights reserved."
__credits__ = "Acceptto Confidential"
__maintainer__ = "Adriana Costa"
__email__ = "adriana.costa@acceptto.com"
__date__ = "2020-05-28"

import logging
import sys
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Code from pyloa
class FeatureSelector:
    """
    Class that will perform feature selection as a preprocessing step.
    
    """

    def __init__(self, data):

        self.data = data
        self.base_features = data.columns

        self.categorical_features = None
        self.numerical_features = None
        self.time_feature = None

        self.missings_threshold = 0.3
        self.missings = None
        self.single_uniques = None

        self.selected_features = None

    def get_col_dtype(self, col):
        """
        Infer datatype of a pandas column. 

        col: a pandas Series representing a df column. 
        """
        col.dropna(inplace=True)
        try:
            col.infer_objects().dtypes == "datetime64[ns, UTC]"
            return "time"
        except:
            try:
                pd.to_numeric(col)
                if np.array_equal(col, col.astype(int)):
                    return "cat"
                else:
                    return "num"
            except:
                return "cat"

    def get_data_dtypes(self):
        """
        Infer datatypes of data frame columns. 

        """
        data_dtypes = np.array(
            [
                [col, self.get_col_dtype(self.data.loc[:, col])]
                for col in self.data.columns
            ]
        )

        self.categorical_features = data_dtypes[
            np.where(data_dtypes[:, 1] == "cat")[0], 0
        ]
        self.numerical_features = data_dtypes[
            np.where(data_dtypes[:, 1] == "num")[0], 0
        ]
        self.time_feature = data_dtypes[np.where(data_dtypes[:, 1] == "time")[0], 0]

    def identify_missings(self):
        """
        Return features that have a missing rate higher than threshold.

        """

        missings = (
            self.data.isnull().sum(axis=0) / len(self.data) > self.missings_threshold
        )

        self.missings = np.array(missings.index[missings])

        if len(self.missings) == 0:
            self.missings = np.array([""])

    def identify_single_unique(self):
        """
        Return categorical features that have a unique value.

        """
        if len(self.categorical_features) == 0:
            self.single_uniques = np.array([""])
        else:
            singles = self.data.loc[:, self.categorical_features].nunique() == 1
            self.single_uniques = np.array(singles.index[singles])

    def select(self):
        """
        Selects features on DataFrame based on missing rate and number of categories.
        """
        self.get_data_dtypes()
        self.identify_missings()
        self.identify_single_unique()

        features_to_remove = np.unique(
            np.concatenate((self.single_uniques, self.missings))
        )

        self.categorical_features = np.array(
            list(
                set(self.categorical_features).symmetric_difference(
                    np.intersect1d(self.categorical_features, features_to_remove)
                )
            )
        )
        self.numerical_features = np.array(
            list(
                set(self.numerical_features).symmetric_difference(
                    np.intersect1d(self.numerical_features, features_to_remove)
                )
            )
        )
        self.time_feature = np.array(
            list(
                set(self.time_feature).symmetric_difference(
                    np.intersect1d(self.time_feature, features_to_remove)
                )
            )
        )

    def get_categories(self, col):
        """
        Returns categories for a specific categorical feature.
        
        """
        if col not in self.categorical_features:
            logger.warning("Not a valid column.")
        else:
            return np.sort(self.data.loc[:, col].dropna().unique())



def get_column_transformer(fs, data):
    """
    This method returns a ColumnTransformer object based on the columns of a FeatureSelector.
    
    """

    time_column = fs.time_feature
    numerical_columns = fs.numerical_features
    categorical_columns = fs.categorical_features

    # initialize column_transformer
    column_transformer = make_column_transformer((FunctionTransformer(), "init"))

    transformers = []
    index = {}
    aux_index = 0

    # temporal features
    if len(time_column) > 0:

        time_column_name = str(time_column[0])

        # time of day
        transformers.append(
            (
                time_column_name + "_num_transformer",
                FunctionTransformer(
                    lambda x: np.array(
                        [
                            pd.Timestamp(ts).hour
                            + pd.Timestamp(ts).minute / 60
                            + pd.Timestamp(ts).second / 3600
                            for ts in x
                        ]
                    ).reshape(-1, 1),
                ),
                time_column_name,
            )
        )
        index["numerical_time"] = [0]

        #  weekday
        transformers.append(
            (
                time_column_name + "_cat_transformer",
                FunctionTransformer(
                    lambda x: np.array(
                        [pd.Timestamp(ts).weekday() for ts in x]
                    ).reshape(-1, 1),
                ),
                time_column_name,
            )
        )
        index["categorical_time"] = [1]
        aux_index = 2

    else:
        index["numerical_time"] = []
        index["categorical_time"] = []

    # numerical features
    for numerical_column_name in numerical_columns:
        transformers.append(
            (
                numerical_column_name + "_transformer",
                FunctionTransformer(lambda x: np.array(x).reshape(-1, 1).astype(np.float64)),
                str(numerical_column_name),
            )
        )
    index["numerical"] = np.arange(
        aux_index, aux_index + len(numerical_columns)
    ).tolist()

    # categorical features
    for categorical_column_name in fs.categorical_features:
        transformers.append(
            (
                categorical_column_name + "_transformer",
                OrdinalEncoder(
                    categories=fs.get_categories(categorical_column_name)[
                        np.newaxis
                    ].tolist()
                ),
                [str(categorical_column_name)],
            )
        )
    index["categorical"] = np.arange(
        aux_index + len(numerical_columns),
        aux_index + len(numerical_columns) + len(categorical_columns),
    ).tolist()

    # update transformers
    column_transformer.transformers = transformers

    return column_transformer, index

def split_anomaly_dataset_(data, column, drop_na=False):
    """
    This method splits the dataset into train and test based on a column .
    
    For instance, I want to train my model with all the approved request but I would like to test it also on the rejected requests.
    """

    data_approved = data[data[column] == "approved"]
    data_rejected = data[data[column] == "rejected"]

    X_train, X_test = train_test_split(data_approved, test_size=0.2)
    y_train = X_train[column]

    y_test = pd.concat([X_test[column], data_rejected[column]])
    X_test = pd.concat([X_test, data_rejected])

    X_train = X_train.drop(columns=[column])
    X_test = X_test.drop(columns=[column])

    if drop_na:
        idx_to_drop_train = X_train.index[X_train.isnull().any(axis=1)].tolist()
        X_train = X_train.drop(index=idx_to_drop_train)
        y_train = y_train.drop(index=idx_to_drop_train)

        idx_to_drop_test = X_test.index[X_test.isnull().any(axis=1)].tolist()
        X_test = X_test.drop(index=idx_to_drop_test)
        y_test = y_test.drop(index=idx_to_drop_test)

    return X_train, X_test, y_train, y_test