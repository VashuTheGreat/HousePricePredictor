import sys
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
# from src.components.model_evaluation import ModelEvaluation
# from src.components.model_pusher import ModelPusher

from src.entity.config_entity import (DataIngestionConfig,
                                          DataValidationConfig,
                                          DataTransformationConfig,
                                          ModelTrainerConfig,
                                        #   ModelEvaluationConfig,
                                        #   ModelPusherConfig
                                          )
                                          
from src.entity.artifact_entity import (DataIngestionArtifact,
                                            DataValidationArtifact,
                                            DataTransformationArtifact,
                                            ModelTrainerArtifact,
                                            # ModelEvaluationArtifact,
                                            # ModelPusherArtifact
                                            )



class Training_pipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        # self.model_evaluation_config = ModelEvaluationConfig()
        # self.model_pusher_config = ModelPusherConfig()


    
    async def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from mongodb")
            data_ingestion =DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact =await data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    async def start_data_validation(self,data_ingestion_artifact: DataIngestionArtifact)->DataValidationArtifact:
        try:
            logging.info("Entered the start_data_validation method of TrainPipeline class")
            logging.info("Validating The Data")
            data_validation =DataValidation()
            await data_validation.init_config(data_validation_config=self.data_validation_config,data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact=await data_validation.initiate_data_validation()
            logging.info("Data Validation Completed")
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
        
    async def start_data_transformation(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        try:
            logging.info("Entererd the start_data_transformation method of TrainingPipeline Class")
            logging.info("transforming the data")
            data_transformation=DataTransformation()
            await data_transformation.init_config(data_ingestion_artifact=data_ingestion_artifact,data_transformation_config=self.data_transformation_config,data_validation_artifact=data_validation_artifact)
            data_transformation_artifact=await data_transformation.initiate_data_transformation()
            logging.info("Data Transformation Completed")
            logging.info("Exited the start_data_transformation method of TrainingPipeline")
            return data_transformation_artifact
        except Exception as e:
            raise MyException(e,sys)
        

    async def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            logging.info("Entered the start_model_trainer method of Training Pipeline")
            logging.info("training the model")
            model_trainer=ModelTrainer()
            await model_trainer.init_config(data_transformation_artifact=data_transformation_artifact,model_trainer_config=self.model_trainer_config)
            model_trainer_artifact=await model_trainer.initiate_model_trainer()
            logging.info("Model trainer Complete")
            logging.info("Exited the sart_model_trainer method of TrainingPipeline")
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e,sys)    
        
    
