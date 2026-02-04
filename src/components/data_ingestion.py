import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
import logging
from src.data_access.proj1_data import Proj1Data

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)

    async def export_data_into_feature_store(self)->DataFrame:
        try:
            proj1=Proj1Data()
            await proj1.connect()
            logging.info("In data ingestion fetching data from proj1")
            dataframe=await proj1.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            feature_file_path=self.data_ingestion_config.feature_store_file_path

            dir_path=os.path.dirname(feature_file_path)

            os.makedirs(dir_path,exist_ok=True)

            logging.info(f"Saving dataframe to features folder of size: {dataframe.shape}")
            dataframe.to_csv(feature_file_path)
            return dataframe
        except Exception as e:
            raise MyException(e,sys)

    async def split_data_as_train_test(self,dataframe:DataFrame)->None:
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")
        try:
            train_test_ratio=self.data_ingestion_config.train_test_split_ratio
            train,test=train_test_split(dataframe,random_state=42,shuffle=True,test_size=train_test_ratio)
            logging.info("Performed train test split on the dataframe")

            dir_path=os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)

            logging.info(f"Exporting train and test file path.")
            train.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)
            logging.info(f"Exported train and test file path.")

        except Exception as e:
            raise MyException(e,sys)

    async def initiate_data_ingestion(self)->DataIngestionArtifact:
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion Class")
        try:
            dataframe=await self.export_data_into_feature_store()
            logging.info("Got the data from mongodb")
            await self.split_data_as_train_test(dataframe)
            logging.info("Performed train test split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e    

                    