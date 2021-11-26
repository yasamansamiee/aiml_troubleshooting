#####################################################################
# Data preprocessing
#####################################################################
# data = data_new.copy()

import json
import re
import pandas as pd
import logging 
from datetime import datetime, timedelta
from .data_reader import Data


logging.basicConfig(
    format="[%(levelname)s|%(name)s|%(funcName)s:%(lineno)d] %(message)s",
    level=logging.WARNING,
)

for name in logging.root.manager.loggerDict:
    if name.startswith("kuberspatiotemporal"):
        logging.getLogger(name).setLevel(logging.DEBUG)

logger = logging.getLogger("jupyter")
logger.setLevel(logging.DEBUG)

class Cleaning(Data):

    def __init__(self):
        Data.__init__(self)
        self.user_name_cleaning()
        self.feature_cleaning()

    def user_name_cleaning(self):
        
        self.data['user_id'] = self.data['user_id'].apply(str)
        self.data['organization_id'] = self.data['organization_id'].apply(str)
        self.data['org_user'] = self.data['organization_id'] + '_' + self.data['user_id']
        self.data['user_id'] = self.data['user_id'].apply(int)
        self.data['organization_id'] = self.data['organization_id'].apply(int)
        self.data["tuple_user_id"] = self.data["org_user"].apply(lambda x : tuple([int(_) for _ in x.split("_")]))
        filtered_data = (self.data.groupby(["org_user", "tuple_user_id"])["event_time"].count()
                            .reset_index()
                            .rename(columns={"event_time": "count"}))
        #uba_minimum_records = int(os.environ.get("uba_minimum_records", 10))
        uba_minimum_records = self.uba_minimum_records
        filtered_users = filtered_data[(filtered_data['count'] > uba_minimum_records)]

        org_user_list = list(filtered_users["org_user"])
    
    def feature_cleaning(self):
        ###only related to BT
        self.data = self.data.rename(
        columns={
            "data_eguardian_dbfp_payload_hash1": "data_eguardian_dbfp_payload_struct_hash1"
        }
        )
        cc = self.data.columns
        print([cc[i] for i in range(len(cc)) if "hash1" in cc[i]])
        self.data = self.data.dropna(
        subset=[
            "data_eguardian_dbfp_payload_struct_hash1", "event_time"],
        )
        self.data["event_time"] = self.data["event_time"].apply(lambda x: datetime.utcfromtimestamp(x))
        self.data["event_time"] = self.data["event_time"].dt.tz_localize("UTC")

        self.data["weekday"] = self.data["event_time"].apply(lambda x: "day" + str(x.weekday()))
        self.data = self.data.dropna(subset=["event_time", "data_eguardian_dbfp_payload_struct_hash1"])
        ### only related to BT
