import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

import mlflow.pyfunc
from fastapi import Depends, FastAPI, HTTPException, status
from mlflow.tracking import MlflowClient

from src.api.predictor import predict
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """Manages MLflow model loading and health state."""

    def __init__(self):
        self.model = None
        self.model_version_info = None
        self.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        self.model_name = "churn_prediction_model"
        self.stage = "Production"

    def load(self):
        """Load the model with retries."""
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient()

        max_retries = 5
        for i in range(max_retries):
            try:
                logger.info(
                    f"Attempting to load model '{self.model_name}' from {self.stage} stage (Attempt {i+1})..."
                )
                model_uri = f"models:/{self.model_name}/{self.stage}"
                self.model = mlflow.pyfunc.load_model(model_uri)

                # Cache version info for /model_info
                versions = client.get_latest_versions(
                    self.model_name, stages=[self.stage]
                )
                if versions:
                    self.model_version_info = versions[0]

                logger.info(f"Model loaded successfully from {model_uri}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                if i < max_retries - 1:
                    time.sleep(10)
        return False

    def get_model(self):
        if not self.model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded",
            )
        return self.model


model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application."""
    logger.info("Starting API lifespan...")
    success = model_manager.load()
    if not success:
        logger.error(
            "API starting without a loaded model. Prediction endpoints will be unavailable."
        )
    yield
    logger.info("Shutting down API lifespan...")


app = FastAPI(title="Churn Prediction API", version="2.0", lifespan=lifespan)


# Dependency to get the model
def get_model_instance():
    return model_manager.get_model()


@app.get("/health")
def health_check():
    """Comprehensive health check."""
    health_status = {
        "status": "ok",
        "model_loaded": model_manager.model is not None,
        "mlflow_connection": False,
    }

    try:
        mlflow.set_tracking_uri(model_manager.tracking_uri)
        mlflow.search_experiments()
        health_status["mlflow_connection"] = True
    except ValueError:
        health_status["status"] = "degraded"

    if not health_status["model_loaded"]:
        health_status["status"] = "degraded"

    return health_status


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(
    request: PredictRequest,
    model: Annotated[mlflow.pyfunc.PyFuncModel, Depends(get_model_instance)],
):
    input_data = request.model_dump()
    prediction, probability = predict(model, input_data)
    return PredictResponse(churn_risk_score=prediction, probability=probability)


@app.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict_endpoint(
    request: BatchPredictRequest,
    model: Annotated[mlflow.pyfunc.PyFuncModel, Depends(get_model_instance)],
):
    input_data = [item.model_dump() for item in request.inputs]
    predictions = []
    for data in input_data:
        prediction, probability = predict(model, data)
        predictions.append(
            PredictResponse(churn_risk_score=prediction, probability=probability)
        )
    return BatchPredictResponse(predictions=predictions)


@app.get("/model_info", response_model=ModelInfoResponse)
def model_info():
    if not model_manager.model_version_info:
        raise HTTPException(status_code=404, detail="Model metadata not available")

    client = MlflowClient()
    version_info = model_manager.model_version_info
    tags = client.get_model_version_tags(
        name=version_info.name, version=version_info.version
    )

    return ModelInfoResponse(
        model_name=version_info.name,
        stage=version_info.current_stage,
        version=version_info.version,
        run_id=version_info.run_id,
        roc_auc=float(tags.get("roc_auc", 0.0)),
    )
