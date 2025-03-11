from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from src.helpers.config import APP_NAME,VERSION,API_SECRET_KEY
from src.controllers.ModelTrainer import ModelTrainer
from src.models.request import TrainingData,TestingData,QueryText
from src.models.response import StatusObject, PredictionObject, PredictionsObject

trainer = ModelTrainer()
app = FastAPI(title=APP_NAME,version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_methods = ['*'],
    allow_headers = ['*'],
)

# Authorization
api_key_header = APIKeyHeader(name='X-API-KEY')
async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_SECRET_KEY:
        raise HTTPException("You can access this api",status_code=403)
    return api_key

# checking for health of api
@app.get('/',tags=["Healthy"],description="Healthy check")
async def home(api_key:str =Depends(verify_api_key)):
    return{
        "app_name":APP_NAME,
        "version":VERSION
    }
    
    
@app.get('/status',tags=["status"],description="Get status")
async def get_status(api_key:str = Depends(verify_api_key)):
    status  = trainer.get_status()
    return StatusObject(**status)


# Training
@app.post("/train", tags=["Training"], description="Train a new model")
def train(training_data: TrainingData, api_key: str=Depends(verify_api_key)):
    try:
        trainer.train(texts=training_data.texts, labels=training_data.Labels)
        status = trainer.get_status()
        return StatusObject(**status)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", tags=["Prediction"], description="Predict single input")
def predict(query_text: QueryText, api_key: str=Depends(verify_api_key)):
    try:
        prediction = trainer.predict(texts=[query_text.text])[0]
        return PredictionObject(**prediction)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict-batch", tags=["Prediction"], description="predict a batch of sentences")
def predict_batch(testing_data: TestingData, api_key: str=Depends(verify_api_key)):
    try:
        predictions = trainer.predict(texts=testing_data.texts)
        return PredictionsObject(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))