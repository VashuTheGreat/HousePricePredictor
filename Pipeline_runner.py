from src.logger import *
import asyncio
from src.exception import MyException
from src.entity.artifact_entity import (RunTrainingArtifact,
                                        DataIngestionArtifact,
                                        DataValidationArtifact,
                                        DataTransformationArtifact,
                                        ModelTrainerArtifact,
                                        RunTrainingArtifact)
import sys
from src.utils.main_utils import write_yaml_file
from src.constants import RUN_TRAINING_DIR_NAME,RUN_TRAINING_FILE_PATH,TRANSFORMED_OBJECT_FILE_PATH,MODEL_FILE_NAME
from src.components.model_evaluation import ModelEvaluation
from dotenv import load_dotenv
load_dotenv()
from src.pipeline.training_pipeline import Training_pipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.main_utils import load_object
from src.components.data_transformation import DataTransformation
import pandas as pd


class RunPipeline:
    def __init__(self,pipeline_name=None):
       pass


    async def run_training(self):
        try:
            
            pipeline=Training_pipeline()
            data_ingestion_artifact=await pipeline.start_data_ingestion()
            data_validation_artifact=await pipeline.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformed_artifact=await pipeline.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact=data_validation_artifact)
            model_trained_artifact=await pipeline.start_model_trainer(data_transformation_artifact=data_transformed_artifact)

            run_training_artifact= RunTrainingArtifact(data_ingestion_artifact=data_ingestion_artifact,
                                       data_validation_artifact=data_transformed_artifact,
                                       data_transformed_artifact=data_transformed_artifact,
                                       model_trained_artifact=model_trained_artifact)
            evaluator=ModelEvaluation()
            await evaluator.init_config(run_train_artifact=run_training_artifact)
            await evaluator.save_model()                      
            
            return run_training_artifact

        except Exception as e:
            raise MyException(e,sys)    
        
    async def run_prediction(self,city:str,area:float,beds:int,bathrooms:int,balconies:int,furnishing:str,area_rate:float,bhk:int):
        dataframe=pd.DataFrame({
            "city":[city],
            "area":[area],
            "beds":[beds],
            "bathrooms":[bathrooms],
            "balconies":[balconies],
            "furnishing":[furnishing],
            "area_rate":[area_rate],
            "bhk":[bhk]
        })
        pipeline=PredictionPipeline()
        
        preprocessor_path = os.path.join(RUN_TRAINING_DIR_NAME,TRANSFORMED_OBJECT_FILE_PATH)
        model_path = os.path.join(RUN_TRAINING_DIR_NAME,MODEL_FILE_NAME)
        
        if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
            return "Model or Preprocessor not found. Please train the model first by clicking 'Train Model'."
            
        preprocessing_object=await load_object(file_path=preprocessor_path)
        model=await load_object(file_path=model_path)
        
        # model is already an instance of MyModel which has its own predict method
        # and contains its own preprocessing_object.
        return await model.predict(dataframe=dataframe)




           



        