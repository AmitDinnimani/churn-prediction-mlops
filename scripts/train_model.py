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
# ----------------

TARGET = "churn_risk_score"
TEST_SIZE = 0.2
RANDOM_STATE = 42

models = {
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(eval_metric="logloss"),
}

# ----------------

df = load_data(DATA_PATH)
is_raw_data_vaild, _ = raw_data_validation(df)

if is_raw_data_vaild:
    df = preprocess_df(df)
    logger.info("Raw Data Vaildation Passed")
else:
    logger.error("Raw Data Validation Failed")
    raise ValueError("Raw Data Validation Failed")

is_processed_data_vaild, _ = processed_data_validation(df)

if not is_processed_data_vaild:
    logger.error("Processed Data Vaildation Failed")
    raise ValueError("Processed Data Vaildation Failed")
logger.info("Processed Data Vaildation Passed")

# ----------------

X = df.drop(columns=[TARGET])
Y = df[TARGET]

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
logger.info("Train Test Split Done")

# ----------------

best_auc = -1
best_run_id = None
best_model_name = None

# ----------------
for name, model in models.items():
    logger.info(f"Starting training for {name}")
    with mlflow.start_run(run_name=name) as run:
        trained_model = train_model(model, x_train, y_train)

        y_pred = trained_model.predict(x_test)
        y_proba = trained_model.predict_proba(x_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_proba)

        mlflow.log_param("model_name", name)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(trained_model, artifact_path="model")

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_run_id = run.info.run_id
            best_model_name = name

        logger.info(f"Finished training {name} | Metrics: {metrics}")

logger.info("All models trained and logged to MLflow successfully!")

# ----------------

register_model(best_run_id, best_model_name, best_auc)
