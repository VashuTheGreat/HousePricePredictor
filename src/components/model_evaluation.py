from src.entity.artifact_entity import RunTrainingArtifact
from src.constants import FINAL_MODEL_FILE_PATH,FINAL_MODEL_PERFORMANCE_PATH,RUN_TRAINING_DIR_NAME,TRANSFORMED_OBJECT_FILE_PATH
from src.utils.main_utils import load_object,save_object,read_yaml_file,write_yaml_file
import logging
from src.exception import MyException
import sys
import os
class ModelEvaluation:
    def __init__(self):
        pass

    async def init_config(self,run_train_artifact:RunTrainingArtifact):
        self.run_train_artifact=run_train_artifact


    async def check_validation(self,per_yaml,to_yaml)->bool:
        if per_yaml['model_trained_artifact']['metric_artifact']['r2_score']>=to_yaml.model_trained_artifact.metric_artifact.r2_score:
            return False
        return True 

    async def save_model(self):
        try:
            model=await load_object(file_path=self.run_train_artifact.model_trained_artifact.trained_model_file_path)
            
            # Check if existing model exists before loading
            if not os.path.exists(FINAL_MODEL_FILE_PATH):
                logging.info(f"Existing model not found at {FINAL_MODEL_FILE_PATH}. Saving first model.")
                await save_object(file_path=FINAL_MODEL_FILE_PATH,obj=model)
                obj=await load_object(file_path=self.run_train_artifact.data_transformed_artifact.transformed_object_file_path)
                await save_object(file_path=os.path.join(RUN_TRAINING_DIR_NAME,TRANSFORMED_OBJECT_FILE_PATH),obj=obj)
                # Also save the performance yaml for first run
                await write_yaml_file(file_path=FINAL_MODEL_PERFORMANCE_PATH,content=self.run_train_artifact)
                return

            saved_model=await load_object(file_path=FINAL_MODEL_FILE_PATH)
            if not saved_model:
                await save_object(file_path=FINAL_MODEL_FILE_PATH,obj=model)
                obj=await load_object(file_path=self.run_train_artifact.data_transformed_artifact.transformed_object_file_path)
                await save_object(file_path=os.path.join(RUN_TRAINING_DIR_NAME,TRANSFORMED_OBJECT_FILE_PATH),obj=obj)
   
            else:
                per_yaml=await read_yaml_file(file_path=FINAL_MODEL_PERFORMANCE_PATH)

                is_ok=await self.check_validation(per_yaml=per_yaml,to_yaml=self.run_train_artifact)

                if is_ok:
                    # saving new model
                    await write_yaml_file(file_path=FINAL_MODEL_PERFORMANCE_PATH,content=self.run_train_artifact)
                    await save_object(file_path=FINAL_MODEL_FILE_PATH,obj=model)
                    obj=await load_object(file_path=self.run_train_artifact.data_transformed_artifact.transformed_object_file_path)
                    await save_object(file_path=os.path.join(RUN_TRAINING_DIR_NAME,TRANSFORMED_OBJECT_FILE_PATH),obj=obj)
        except Exception as e:
            raise MyException(e,sys)            




