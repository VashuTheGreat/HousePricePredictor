from fastapi import FastAPI, Request, Form, Body
from src.utils.main_utils import read_yaml_file, write_yaml_file
from src.constants import MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn as uv
from src.logger import *
from Pipeline_runner import RunPipeline
import shutil

pipeline=RunPipeline()
app=FastAPI(title="House Rent Predictor",version="0.0.1")

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    

@app.post("/predict")
async def predict(city: str = Form(...), 
                  area: float = Form(...), 
                  beds: int = Form(...), 
                  bathrooms: int = Form(...), 
                  balconies: int = Form(...), 
                  furnishing: str = Form(...), 
                  area_rate: float = Form(...),
                  bhk:int=Form(...)):
                  
    res = await pipeline.run_prediction(city, area, beds, bathrooms, balconies, furnishing, area_rate,bhk)
    logging.info(f"Prediction result type: {type(res)}")
    if isinstance(res, str):
        return res
    # Ensure we get a serializable value
    if hasattr(res, "__iter__"):
        return float(res[0])
    return float(res)


@app.get("/train")
async def train(password: str = None):
    
    if password != "a":
        return {"detail": "Only admin has access to retrain the model"}, 403
    shutil.rmtree("artifact",ignore_errors=True)
    shutil.rmtree("logs",ignore_errors=True)    
    res=await pipeline.run_training()
    return res

@app.get("/experiment")
async def experiment(request: Request):
    return templates.TemplateResponse("experiment.html", {"request": request})

@app.get("/get_config")
async def get_config():
    try:
        config = await read_yaml_file(file_path=MODEL_TRAINER_MODEL_CONFIG_FILE_PATH)
        return config
    except Exception as e:
        return {"error": str(e)}, 500

@app.post("/update_config")
async def update_config(config: dict = Body(...)):
    try:
        await write_yaml_file(file_path=MODEL_TRAINER_MODEL_CONFIG_FILE_PATH, content=config)
        return {"message": "Configuration updated successfully"}
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    uv.run("main:app",port=8000,reload=True)
