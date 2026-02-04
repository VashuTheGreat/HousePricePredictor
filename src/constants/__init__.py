import os
from datetime import date
from src.utils.main_utils import read_yaml_file_sync

config=read_yaml_file_sync(os.path.join("config","model.yaml"))

COLLECTION_NAME = "COLLECTION_NAME"
MONGODB_URL_KEY = "MONGODB_URL"
DATABASE_NAME_KEY="DATABASE_NAME_KEY"

PIPELINE_NAME: str = "training_pipeline"
ARTIFACT_DIR: str = "artifact"
DATA_INGESTION_DIR_NAME="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR="features"
FILE_NAME: str = "data.csv"
DATA_INGESTION_INGESTED_DIR:str="ingested"
TRAIN_FILE_NAME:str="train.csv"
TEST_FILE_NAME:str="test.csv"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float=config["_train_test_split_ratio"]
DATA_INGESTION_COLLECTION_NAME="Proj1-Data"
DATA_VALIDATION_DIR_NAME="data_validation"
DATA_VALIDATION_REPORT_FILE_NAME="validation_report.yaml"
SCHEMA_FILE_PATH=os.path.join("config","schema.yaml")

DATA_TRANSFORMATION_DIR="data_transformed"
TRANSFORMED_TRAIN_FILE_PATH="train.npy"
TRANSFORMED_TEST_FILE_PATH="test.npy"
TRANSFORMED_OBJECT_FILE_PATH="Objects"
BHK_RE_EXTRACTOR="(\d+)\s*BHK"

SCHEMA_MODEL_FILE_PATH=os.path.join("config","model.yaml")

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_FILE_NAME="model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = config["_model_trainer_expected_score"]
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
MODEL_TRAINER_N_ESTIMATORS=config["_n_estimators"]
MODEL_TRAINER_MIN_SAMPLES_SPLIT: int = config["_min_samples_split"]
MODEL_TRAINER_MIN_SAMPLES_LEAF: int = config["_min_samples_leaf"]
MIN_SAMPLES_SPLIT_MAX_DEPTH: int = config["_max_depth"]
MIN_SAMPLES_SPLIT_CRITERION: str = 'squared_error'
MIN_SAMPLES_SPLIT_RANDOM_STATE: int = config["_random_state"]


RUN_TRAINING_ARTIFACT_FILE_PATH=os.path.join("artifact",)
RUN_TRAINING_DIR_NAME="saved_model"
RUN_TRAINING_FILE_PATH="run_training.yaml"
FINAL_MODEL_FILE_PATH=os.path.join("saved_model","model.pkl")
FINAL_MODEL_PERFORMANCE_PATH=os.path.join("saved_model","run_training.yaml")
