import os

import mlflow.pyfunc
from fastapi import FastAPI
from mlflow.tracking import MlflowClient

from src.api.predictor import predict
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from src.utils.logger import configure_logging

configure_logging()


mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
model = mlflow.pyfunc.load_model("models:/churn_prediction_model/Production")


app = FastAPI(title="Churn Prediction API", version="1.0")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    input_data = request.model_dump()
    prediction, probability = predict(model, input_data)
    return PredictResponse(churn_risk_score=prediction, probability=probability)


@app.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict_endpoint(request: BatchPredictRequest):
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
    client = MlflowClient()

    # Get the model versions in 'Production'
    versions = client.get_latest_versions(
        name="churn_prediction_model", stages=["Production"]
    )
    if not versions:
        raise ValueError("No model in Production stage")

    version_info = versions[0]  # Take the latest production model
    tags = client.get_model_version_tags(
        name="churn_prediction_model", version=version_info.version
    )

    return ModelInfoResponse(
        model_name=version_info.name,
        stage=version_info.current_stage,
        version=version_info.version,
        run_id=version_info.run_id,
        roc_auc=float(tags.get("roc_auc", 0.0)),
    )
