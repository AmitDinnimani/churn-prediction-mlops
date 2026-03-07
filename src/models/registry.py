import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_mlflow_client():
    """Lazily initialize and return an MlflowClient."""
    return MlflowClient()


def register_model(best_run_id, best_model_name, best_auc):
    """Register the best model and transition it to Production."""
    logger.info(
        f"Registering best model: {best_model_name} with ROC-AUC={best_auc:.4f}"
    )

    try:
        client = get_mlflow_client()

        # Register the model
        model_uri = f"runs:/{best_run_id}/model"
        model_name = "churn_prediction_model"

        result = mlflow.register_model(model_uri=model_uri, name=model_name)

        # Transition to Production
        client.transition_model_version_stage(
            name=model_name, version=result.version, stage="Production"
        )
        logger.info(
            f"Successfully registered model version {result.version} in Production."
        )
        return result
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise
