import sys
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact,RegressionMetricArtifact
from src.entity.estimator import MyModel
import mlflow
import mlflow.sklearn

class ModelTrainer:
    def __init__(self):
        pass

    async def init_config(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    async def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, dict]:
        try:
            logging.info("Training RandomForestRegressor with specified parameters")

            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

            logging.info("Train-test split done.")

            model = RandomForestRegressor(
                n_estimators=self.model_trainer_config.n_estimators,
                min_samples_split=self.model_trainer_config.min_samples_split,
                min_samples_leaf=self.model_trainer_config.min_samples_leaf,
                max_depth=self.model_trainer_config.max_depth,
                criterion=self.model_trainer_config.criterion,
                random_state=self.model_trainer_config.random_state
            )

            logging.info("Model training started...")
            model.fit(x_train, y_train)
            logging.info("Model training completed.")

            # Predictions and evaluation metrics
            y_pred = model.predict(x_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            metric_artifact = RegressionMetricArtifact(r2_score=r2, mse=mse, mae=mae)


            

            # MLflow logging
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)

            mlflow.log_param("n_estimators", self.model_trainer_config.n_estimators)
            mlflow.log_param("min_samples_split", self.model_trainer_config.min_samples_split)
            mlflow.log_param("min_samples_leaf", self.model_trainer_config.min_samples_leaf)
            mlflow.log_param("max_depth", self.model_trainer_config.max_depth)
            mlflow.log_param("criterion", self.model_trainer_config.criterion)
            mlflow.log_param("random_state", self.model_trainer_config.random_state)

            mlflow.sklearn.log_model(model, artifact_path="model")

            return model, metric_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    async def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Started Model Training Component")

            # Load transformed train-test data
            train_arr = await load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = await load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )
            logging.info("Train-test data loaded successfully.")

            # Train model and get metrics
            trained_model, metric_artifact = await self.get_model_object_and_report(
                train=train_arr, test=test_arr
            )
            logging.info("Model object and metrics obtained.")

            # Load preprocessing object
            preprocessing_obj = await load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            logging.info("Preprocessing object loaded.")

            logging.info("Saving new model as performance is better than previous one.")
            my_model = MyModel()
            await my_model.init_config(
                preprocessing_object=preprocessing_obj, trained_model_object=trained_model
            )
            await save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object including preprocessing and trained model.")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e
