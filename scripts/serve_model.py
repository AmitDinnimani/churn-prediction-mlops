import mlflow.pyfunc

mlflow.set_tracking_uri("http://mlflow:5000")


def load_model_for_prediction():
    model = mlflow.pyfunc.load_model("models:/churn_prediction_model/Production")

    return model
