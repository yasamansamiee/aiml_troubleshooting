from functools import lru_cache
from typing import List, Mapping
from pydantic import BaseSettings
from pyathena import connect
import pandas as pd

class HandlerSettings(BaseSettings):
    redis_hosts: List[Mapping[str, str]] = None
    query_result_location: str = "s3://datahub-data-store-dev/athena_query_results/"
    model_store_location: str = "s3://datahub-model-store-dev/models/kt/cyber"
    database_name: str = "datahub"
    table_name: str = "cyber"
    max_row_limit: int = 3000
    region_name: str = "us-east-1"
    log_level: str = "WARNING"
    columns_select_statement: str = "from_unixtime(CAST(timestamp AS BIGINT)) as timestamp, auth_status"
    conjuncted_with_statement: str = "timestamp is not null"




@lru_cache()
def get_settings():
    return HandlerSettings()

class Data():

    def __init__(self, query = True):
        self.model_name = "UBT"
        self.application_name = "OKTA"
        self.data = None
        self.uba_minimum_records = 40
        if query ==True:
             self.query_aws()

    def query_aws(self):

        if self.model_name == "UBT":
            
            athena_query = f'SELECT * FROM "datahub"."dbfp_flattened" \
                        where user_id in (Select user_id FROM "datahub"."dbfp_flattened" group by user_id having count(*)>={self.uba_minimum_records}) \
                        and user_id in (Select distinct user_id FROM "datahub"."dbfp_flattened" \
                                        where application_uid = \'41a7d6a4cea23e62bbaf4ecf7f973e4a441adea917320e3908cb461b30b71708\')'
        conn = connect(
            s3_staging_dir=get_settings().query_result_location,
            region_name=get_settings().region_name,)
        self.data = pd.read_sql(athena_query, conn)
        return self.data

