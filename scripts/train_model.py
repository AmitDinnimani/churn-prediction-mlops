import os
import time

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.loader import load_data
from src.data.preprocessor import preprocess_df
from src.data.validator import processed_data_validation, raw_data_validation
from src.models.evaluate import evaluate_model
from src.models.registry import register_model
from src.models.train import train_model
from src.utils.config import DATA_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChurnTrainingPipeline:
    """
    Enterprise-level training pipeline for Churn Prediction.
    Handles data loading, validation, preprocessing, training, and registration.
    """

    def __init__(self, tracking_uri=None):
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://mlflow:5000"
        )
        self.target = "churn_risk_score"
        self.test_size = 0.2
        self.random_state = 42
        self.models = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "xgboost": XGBClassifier(eval_metric="logloss", random_state=42),
        }

    def setup_mlflow(self):
        """Configure MLflow with retries."""
        logger.info(f"Setting MLflow Tracking URI: {self.tracking_uri}")
        mlflow.set_tracking_uri(self.tracking_uri)
        # Verify connection
        max_retries = 3
        for i in range(max_retries):
            try:
                mlflow.search_experiments()
                logger.info("Successfully connected to MLflow.")
                return True
            except Exception as e:
                logger.warning(f"MLflow connection attempt {i+1} failed: {e}")
                if i < max_retries - 1:
                    time.sleep(5)
        logger.error("Failed to connect to MLflow after multiple attempts.")
        return False

    def run(self):
        """Execute the full training pipeline."""
        start_time = time.time()
        logger.info("Starting Churn Prediction Training Pipeline")

        try:
            if not self.setup_mlflow():
                raise ConnectionError("Could not initialize MLflow tracking.")

            # 1. Load Data
            df = load_data(DATA_PATH)

            # 2. Raw Data Validation
            is_valid, report = raw_data_validation(df)
            if not is_valid:
                logger.error(f"Raw data validation failed: {report}")
                raise ValueError("Aborting pipeline due to invalid raw data.")
            logger.info("Raw data validation passed.")

            # 3. Preprocessing
            X = df.drop(columns=[self.target])
            y = df[self.target]

            preprocessor = preprocess_df(df)

            # Configure steps for DataFrame output
            for name, step in preprocessor.named_steps.items():
                if hasattr(step, "set_output"):
                    step.set_output(transform="pandas")

            X_processed = preprocessor.fit_transform(X)
            logger.info(f"Preprocessing completed. Feature shape: {X_processed.shape}")

            # 4. Processed Data Validation
            is_valid, report = processed_data_validation(X_processed)
            if not is_valid:
                logger.error(f"Processed data validation failed: {report}")
                raise ValueError("Aborting pipeline due to invalid processed data.")
            logger.info("Processed data validation passed.")

            # 5. Train/Test Split
            x_train, x_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=self.test_size, random_state=self.random_state
            )

            # 6. Model Training & Evaluation
            best_auc = -1
            best_run_id = None
            best_model_name = None

            for name, model_obj in self.models.items():
                with mlflow.start_run(run_name=name) as run:
                    logger.info(f"Training {name}...")
                    trained_model = train_model(model_obj, x_train, y_train)

                    y_pred = trained_model.predict(x_test)
                    y_proba = trained_model.predict_proba(x_test)[:, 1]
                    metrics = evaluate_model(y_test, y_pred, y_proba)

                    # Log to MLflow
                    mlflow.log_params(model_obj.get_params())
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(trained_model, artifact_path="model")

                    logger.info(f"Model {name} Metrics: {metrics}")

                    if metrics["roc_auc"] > best_auc:
                        best_auc = metrics["roc_auc"]
                        best_run_id = run.info.run_id
                        best_model_name = name

            # 7. Model Registration
            if best_run_id:
                logger.info(
                    f"Best model found: {best_model_name} (AUC: {best_auc:.4f})"
                )
                register_model(best_run_id, best_model_name, best_auc)
            else:
                logger.warning("No model was eligible for registration.")

            duration = time.time() - start_time
            logger.info(f"Pipeline completed successfully in {duration:.2f} seconds.")

        except Exception as e:
            logger.exception(f"Pipeline failed with an unexpected error: {e}")
            raise


if __name__ == "__main__":
    pipeline = ChurnTrainingPipeline()
    pipeline.run()
