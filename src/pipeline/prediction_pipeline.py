from src.entity.estimator import MyModel
from sklearn.pipeline import Pipeline
import pandas as pd
from src.utils.main_utils import read_yaml_file
from src.constants import SCHEMA_MODEL_FILE_PATH
import logging
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from src.entity.artifact_entity import RegressionMetricArtifact
class PredictionPipeline:
    def __init__(self):
        pass

    async def init_config(self,preprocessing_object: Pipeline, trained_model_object: object):
        self.model=MyModel()
        await self.model.init_config(preprocessing_object=preprocessing_object, trained_model_object=trained_model_object)
        self._schema_model=await read_yaml_file(file_path=SCHEMA_MODEL_FILE_PATH)


    async def predict(self,data:pd.DataFrame):
        return await self.model.predict(dataframe=data)   

    async def start_evaluate_model(self,dataframe:pd.DataFrame):
        logging.info("Entered in the start_evaluate_performance method of Prediction pipeline")
        y_test=dataframe[self._schema_model['target_column']]
        y_pred=await self.model.predict(dataframe=dataframe)
        logging.info("Calculating Performance")
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        metric_artifact = RegressionMetricArtifact(r2_score=r2, mse=mse, mae=mae)
        return metric_artifact
      
             

    