import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from functions import train_model, make_prediction

app = FastAPI(
    title="Machine Learning API",
    description="Cette API permet d'entraîner un modèle de machine learning et de faire des prédictions.",
    version="1.0.0"
)

# CORS middleware
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainingData(BaseModel):
    data: List[List[float]]
    target: List[float]

class PredictionData(BaseModel):
    data: List[List[float]]

@app.post("/training", tags=["Model Training"], summary="Entraîne un modèle sur les données fournies")
async def training(training_data: TrainingData):
    try:
        logger.info("Début de l'entraînement du modèle")
        df = pd.DataFrame(training_data.data)
        target = pd.Series(training_data.target)
        logger.info(f"DataFrame d'entraînement:\n{df.head()}")
        logger.info(f"Target d'entraînement:\n{target.head()}")
        
        model = train_model(df, target)
        model_path = "model/model.joblib"
        
        if not os.path.exists("model"):
            os.makedirs("model")
        
        joblib.dump(model, model_path)
        logger.info(f"Modèle sauvegardé à {model_path}")
        return {"message": "Modèle entraîné et sauvegardé avec succès"}
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du modèle: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'entraînement du modèle: {e}")

@app.post("/predict", tags=["Prediction"], summary="Fait une prédiction avec le modèle entraîné")
async def predict(prediction_data: PredictionData):
    try:
        model_path = "model/model.joblib"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Modèle non trouvé. Entraînez le modèle avant de faire des prédictions.")
        
        model = joblib.load(model_path)
        df = pd.DataFrame(prediction_data.data)
        predictions = make_prediction(model, df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction: {e}")

@app.get("/model", tags=["External API Call"], summary="Appel à une API externe (OpenAI ou HuggingFace)")
async def get_model():
    return {"message": "Cet endpoint fera un appel à une API externe"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
