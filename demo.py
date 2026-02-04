from src.logger import *
import asyncio
from dotenv import load_dotenv
load_dotenv()
from src.pipeline.training_pipeline import Training_pipeline

train_pipe=Training_pipeline()

data_ingestion_artifact=asyncio.run(train_pipe.start_data_ingestion())
# asyncio.run(train_pipe.start_data_validation(data_ingestion_artifact))
data_validation_artifact=asyncio.run(train_pipe.start_data_validation(data_ingestion_artifact))

data_transformed_artifact=asyncio.run(train_pipe.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact=data_validation_artifact))
model_trained_artifact=asyncio.run(train_pipe.start_model_trainer(data_transformation_artifact=data_transformed_artifact))
print(model_trained_artifact)



# -------------- prediction ---------------------------
model_path=




