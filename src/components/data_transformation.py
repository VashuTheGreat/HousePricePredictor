import os
import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from src.utils.main_utils import read_yaml_file, save_numpy_array_data, save_object
from src.constants import SCHEMA_FILE_PATH, BHK_RE_EXTRACTOR
from src.exception import MyException
import sys
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
# handeling imbalance data
import smogn
import numpy as np
import re
from typing import Any

class DataTransformation:
    def __init__(self):
        pass

    async def init_config(self,
                          data_ingestion_artifact: DataIngestionArtifact,
                          data_transformation_config: DataTransformationConfig,
                          data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = await read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)  

    @staticmethod
    async def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    async def get_data_transformer_object(self) -> Pipeline:
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            scaler = StandardScaler()
            encoder = OrdinalEncoder()
            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")

            num_features = self._schema_config['num_features']
            en_features = self._schema_config['encode_columns']

            logging.info("Cols loaded from schema")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", scaler, num_features),
                    ("OrdinalEncoder", encoder, en_features)
                ],
                remainder="passthrough"
            )

            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline
        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method")
            raise MyException(e, sys)
        
    async def _addCol(self, df: pd.DataFrame) -> pd.DataFrame:
        "Adding bhk col"
        def bhk(text):
            match = re.search(BHK_RE_EXTRACTOR, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return 0
        df[self._schema_config['add_columns']['bhk'][0]] = df[self._schema_config['add_columns']['bhk'][1]].apply(bhk)
        return df

    async def _dropCol(self, df: pd.DataFrame) -> pd.DataFrame:
        "Dropping Columns"
        return df.drop(self._schema_config['drop_columns'], axis=1)

    async def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started !!!")
            print(self.data_validation_artifact)
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load Train-Test Data
            train_df = await self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = await self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            # Split Input & Target
            input_feature_train_df = train_df.drop(columns=[self._schema_config['target_column'][0]])
            target_feature_train_df = train_df[self._schema_config['target_column'][0]]

            input_feature_test_df = test_df.drop(columns=[self._schema_config['target_column'][0]])
            target_feature_test_df = test_df[self._schema_config['target_column'][0]]
            logging.info("Input and Target cols defined for both train and test df.")

            # Custom Transformations
            input_feature_train_df = await self._addCol(input_feature_train_df)
            input_feature_train_df = await self._dropCol(input_feature_train_df)

            input_feature_test_df = await self._addCol(input_feature_test_df)
            input_feature_test_df = await self._dropCol(input_feature_test_df)
            logging.info("Custom transformations applied to train and test data")

            # Preprocessing
            logging.info("Starting data transformation")
            preprocessor = await self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end for train-test df.")

            # # ----------------- Apply SMOGN for Regression -----------------
            # logging.info("Applying SMOGN for handling imbalanced regression target")

            # # Ensure target is 1D float and no NaNs
            # target_feature_train_df = target_feature_train_df.astype(float).dropna().reset_index(drop=True)

            # # Combine features + target into single DataFrame
            # train_smogn_df = pd.concat(
            #     [pd.DataFrame(input_feature_train_arr), target_feature_train_df],
            #     axis=1
            # )
            # train_smogn_df.columns = [f"f{i}" for i in range(train_smogn_df.shape[1]-1)] + [self._schema_config['target_column'][0]]

            # # Apply SMOGN (train data only) with explicit k
            # train_smogn_df = smogn.smoter(
            #     data=train_smogn_df,
            #     y=self._schema_config['target_column'][0],
            #     k=5  # explicit neighbors
            # )

            # # Split features & target back
            # input_feature_train_final = train_smogn_df.drop(columns=[self._schema_config['target_column'][0]]).values
            # target_feature_train_final = train_smogn_df[self._schema_config['target_column'][0]].values

            # # Test data: do not resample
            # input_feature_test_final = input_feature_test_arr
            # target_feature_test_final = target_feature_test_df.values
            # logging.info("SMOGN applied to train data. Test data left unchanged.")

            # Concatenate features + target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Feature-target concatenation done for train-test df.")

            # Save preprocessor and transformed data
            await save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            await save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            await save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saved transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e


    @staticmethod
    async def _addCol_pred(data:pd.DataFrame,schema_config:Any) -> pd.DataFrame:
        try:
            def bhk(text):
                match = re.search(BHK_RE_EXTRACTOR, text, re.IGNORECASE)
                if match:
                    return int(match.group(1))
                return 0
            data[schema_config['add_columns']['bhk'][0]] = data[schema_config['add_columns']['bhk'][1]].apply(bhk)
            return data
        except Exception as e:
            raise MyException(e, sys) from e
    @staticmethod
    async def _dropCol_pred(data:pd.DataFrame,schema_config:Any) -> pd.DataFrame:
        try:
            return data.drop(schema_config['drop_columns'], axis=1,errors="ignore")
        except Exception as e:
            raise MyException(e, sys) from e                

    @staticmethod
    async def data_transformation_for_prediction(data:pd.DataFrame,preprocessor:object) -> np.ndarray:
        try:
            logging.info("Data Transformation Started !!!")
    


           

            # Custom Transformations
            schema_config=await read_yaml_file(file_path=SCHEMA_FILE_PATH)
            input_feature_train_df = await DataTransformation._addCol_pred(data,schema_config)
            input_feature_train_df = await DataTransformation._dropCol_pred(input_feature_train_df,schema_config)

            logging.info("Custom transformations applied to data")

            # Preprocessing
            logging.info("Starting data transformation")
            logging.info("Got the preprocessor object")

            input_feature_train_arr = preprocessor.transform(input_feature_train_df)
            logging.info("Transformation done end to end for data.")
            return input_feature_train_arr

        except Exception as e:
            raise MyException(e, sys) from e        
