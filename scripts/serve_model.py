import mlflow.pyfunc
import uvicorn

mlflow.set_tracking_uri("http://mlflow:5000")


def load_model_for_prediction():
    model = mlflow.pyfunc.load_model("models:/churn_prediction_model/Production")

    return model


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000)
