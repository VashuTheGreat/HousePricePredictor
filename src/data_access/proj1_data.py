import sys
import pandas as pd
import numpy as np
from typing import Optional
import logging
from src.configuration.mongo_db_connections import MongoDBClient
from src.constants import DATABASE_NAME_KEY
from src.exception import MyException

class Proj1Data:
    def __init__(self)->None:
        try:
            self.mongo_client=MongoDBClient(database_name=DATABASE_NAME_KEY)

        except Exception as e:
            raise MyException(e,sys)

    async def connect(self):
        logging.info("Connecting to database")
        await self.mongo_client.connect()
    async def export_collection_as_dataframe(self,collection_name:str,database_name: Optional[str] = None)->pd.DataFrame:
        try:
            if database_name is None and collection_name:
                logging.debug(f"collection using collection name {collection_name}")
                collection=self.mongo_client.database[collection_name] 
            else:
                logging.debug(f"collection using datase name {database_name}")
                collection=self.mongo_client[database_name][collection_name]    
            
            logging.debug(f"Collection we found is {collection}")

            logging.info("Fetching Data from MongoDB")
            df=pd.DataFrame(await collection.find().to_list(length=None))
            logging.info("Data Fetched from MongoDB")
            if "id" in df.columns.to_list():
                df = df.drop(columns=["id"], axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        
        except Exception as e:
            raise MyException(e,sys)
