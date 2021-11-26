from collections import namedtuple
import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
import dill
from time import process_time
from sklearn.metrics import confusion_matrix

from kuberspatiotemporal.tools.data import (
    FeatureSelector,
    get_column_transformer,
)
from kuberspatiotemporal import (
    Feature, 
    SpatialModel, 
    KuberModel,
)
from kuberspatiotemporal.lazy_compound import LazyCompoundModel


# Named tuple to help structure results
PipelinesTuple = namedtuple('PipelinesTuple', ['approved', 'rejected'])

##########
# Logging
#
logger = logging.getLogger(__name__)


class TrainState:

    def __init__(
            self,
            approved: Pipeline = None,
            time_approved: float = 0,
            metrics_approved=None,):
        
        self.pipeline_approved = approved
        self.training_time_approved = time_approved
        self.metrics_approved = metrics_approved
        self.pipeline_approved_binary = dill.dumps(approved)


class Train:
    
    def __init__(self, organization_id, user_id):
        self.organization_id = organization_id
        self.user_id = user_id
        self.noise_probability = 0.0005
        self.noise_probability_training = 0.0005
        self.scaling_parameter = 0.7
        
    def train_model(self, origin_data: pd.DataFrame, train_size=0.8) -> TrainState:
        r"""
        The main and public method to be called to train real data
        """

        #approved_test would be used for the cross scoring with label of 1. 
        #rejected would be only for test and would corresspond to the data from other users, therefore we would only have rejected test. 
        # So the approved train and test are the data for this particular user for which the MFA=1
        # Where do we want to input the userid? I put it here but if you think it's better elsewhere just change it. 
        # The test is on OKTA Prod. 
  
        user_id = self.user_id
        organization_id = self.organization_id

        data_approved = origin_data[
            (origin_data["data_eguardian_auth_status"] == "approved")
             & (origin_data["user_id"] == user_id)
             & (origin_data["organization_id"] == organization_id)
        ]

        data_rejected = origin_data[
            (origin_data["data_eguardian_auth_status"] == "approved")
            & (origin_data["user_id"] != user_id)
            #& (origin_data["organization_id"] != organization_id)
        ]

        
        msk_approved = np.random.rand(len(data_approved)) < train_size
        
        if np.sum(~msk_approved) <= len(data_rejected):
            approved_test = pd.concat([data_approved[~msk_approved], data_rejected.sample(n=np.sum(~msk_approved))])
        else:
            approved_test = pd.concat([data_approved[~msk_approved], data_rejected])
        approved = {'train': data_approved[msk_approved], 'test': approved_test}

        return self.train_from_data(approved)


    def train_from_dataframe(self, data: pd.DataFrame, date_field: str = "timestamp", metrics: bool = True) -> tuple:

        fs = FeatureSelector(data["train"][["weekday", "event_time","data_eguardian_dbfp_payload_struct_hash1"]])
        fs.select()
        X_train = data["train"][["weekday", "event_time","data_eguardian_dbfp_payload_struct_hash1"]]
        X_train = X_train[
            np.concatenate(
                (fs.categorical_features,  fs.time_feature)
            )]
        column_transformer, index = get_column_transformer(fs)

        # features model
        features_cpd_model = []
        n_dim = len(fs.time_feature) + len(fs.numerical_features) + len(fs.categorical_features)

        idx_spatial = np.concatenate((index["numerical"], index["numerical_time"])).astype('int')
        idx_kuber = np.concatenate((index["categorical"], index["categorical_time"])).astype('int')

        if len(idx_spatial) > 0:

            spatial_transformed = column_transformer.fit_transform(X_train)[:, idx_spatial]

            limits = np.array(
                [
                    np.min(spatial_transformed, axis=0),
                    np.max(spatial_transformed, axis=0),
                ])

            features_cpd_model.append(
                Feature(
                    SpatialModel(
                        n_dim=len(idx_spatial),
                        min_eigval=1e-10,
                        box = 1, 
                        limits=limits,
                        covar_factor=np.array([np.cov(spatial_transformed[:, i], spatial_transformed[:, i])[
                                            0][0] for i in range(spatial_transformed.shape[1])]),
                    ),
                    idx_spatial,
                ))
        spatial_transformed = column_transformer.fit_transform(X_train)[:, idx_kuber]

        for idx_cat, name in zip(idx_kuber, fs.categorical_features):
            features_cpd_model.append(
                Feature(
                    KuberModel(n_symbols=len(fs.get_categories(name))),
                    [idx_cat],
                ))
        kst = LazyCompoundModel(
        n_dim=np.sum([len(x) for x in index.values()]),
        n_iterations=200,
        scaling_parameter= self.scaling_parameter,
        
        nonparametric=True,
        online_learning=False,
        loa=True,
        features=features_cpd_model,
        noise_probability = self.noise_probability
        , noise_probability_training= self.noise_probability_training, 
        min_samples = 100)
        print('fs.categorical_features', fs.categorical_features) #================================
        print('fs.numerical_features',fs.numerical_features) #=====================================
        print('fs.time_feature', fs.time_feature) #================================================

        pipeline_created = make_pipeline(column_transformer, kst)
        pipeline_created.fit(X_train)

    #     pipeline_created["spatialmodel"].score_threshold = pipeline_created["spatialmodel"].get_score_threshold(
    #         pipeline_created["columntransformer"].transform(data['train']),
    #         lower_quantile=0,
    #         upper_quantile=0.8,
    #     )

        performance_metrics = None
        if metrics:
            performance_metrics = self.test_from_data(data['test'], pipeline_created)

        return (pipeline_created, performance_metrics)


    def train_from_data(self, data_approved: pd.DataFrame) -> TrainState:
        

        time_approved_start = process_time()
        pipeline_approved = None
        metrics_approved = None
        if 'train' in data_approved and not data_approved['train'].empty:
            pipeline_approved, metrics_approved = self.train_from_dataframe(data_approved, metrics=True)

        time_approved_end = process_time()
        time_approved = time_approved_end - time_approved_start

        return TrainState(
            approved=pipeline_approved,
            time_approved=time_approved,
            metrics_approved=metrics_approved,
        )


    def test_from_data(self, test_data: pd.DataFrame, trained_model: Pipeline,threshold = 0.5) -> pd.DataFrame:

        test_approved = test_data[(test_data['user_id'] == self.user_id) 
                                  & (test_data["organization_id"] == self.organization_id)]
        test_rejected = test_data[(test_data['user_id'] != self.user_id) 
                                 # & (test_data["organization_id"] != self.organization_id)
                                 ]


        y_true = np.concatenate(
            (np.ones(len(test_approved)), np.zeros(len(test_rejected)))
        )
        
        y_pred = np.concatenate((np.array([trained_model.score_samples(test_approved.iloc[[idx]]) for idx in range(
            len(test_approved))]), np.array([trained_model.score_samples(test_rejected.iloc[[idx]]) for idx in range(len(test_rejected))])))

        y_pred = [[1]  if _> 0.5 else [0] for _ in y_pred]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        acc = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1_score = 2 * precision * recall / (precision + recall)

        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'specificity': specificity, 'f1_score': f1_score}


    def restore_pipelines(self, pipelines: PipelinesTuple) -> PipelinesTuple:

        approved, rejected = pipelines

        recovered_approved = dill.loads(approved)
        recovered_rejected = dill.loads(rejected)

        return PipelinesTuple(approved=recovered_approved, rejected=recovered_rejected)


