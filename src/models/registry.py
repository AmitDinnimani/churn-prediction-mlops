import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger

logger = get_logger(__name__)
client = MlflowClient()


def register_model(best_run_id, best_model_name, best_auc):
    logger.info(f"Registering best model: {best_model_name} with ROC-AUC={best_auc}")

    result = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/model", name="churn_prediction_model"
    )

    client.transition_model_version_stage(
        name="churn_prediction_model", version=result.version, stage="Production"
    )
