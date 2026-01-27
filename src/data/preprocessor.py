import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger

DROP_COLS = ["security_no", "referral_id", "joining_date", "last_visit_time"]

TARGET = "churn_risk_score"

logger = get_logger(__name__)


def fill_missing_values(df):
    """
    Fill missing values in the dfset.
    """

    try:
        df["region_category"].fillna("Unknown", inplace=True)
        df["preferred_offer_types"].fillna(
            df["preferred_offer_types"].mode()[0], inplace=True
        )
        df["points_in_wallet"].fillna(df["points_in_wallet"].median(), inplace=True)
        logger.info("Missing values filled successfully.")
    except Exception as e:
        logger.error(f"Error in filling missing values: {e}")
        raise e
    return df


def drop_unnecessary_columns(df):
    """
    Drop columns that are not needed for training.
    """
    return df.drop(columns=DROP_COLS, errors="ignore")


def datetime_cols_conversion(df):
    """
    Convert datetime columns to appropriate format.
    """
    df["joining_date"] = pd.to_datetime(df["joining_date"], errors="coerce")
    df["last_visit_time"] = pd.to_datetime(df["last_visit_time"], errors="coerce")

    logger.info("Datetime columns converted successfully.")

    return df


def feature_engineering(df):
    """
    Perform feature engineering on the dfset.
    """

    try:
        for col in ["days_since_last_login", "avg_time_spent"]:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        today = df["last_visit_time"].max()

        df["customer_tenure_days"] = (today - df["joining_date"]).dt.days

        df["last_visit_hour"] = df["last_visit_time"].dt.hour
        df["last_visit_dayofweek"] = df["last_visit_time"].dt.dayofweek

        df["avg_frequency_login_days"] = pd.to_numeric(
            df["avg_frequency_login_days"], errors="coerce"
        )

        df["avg_frequency_login_days"].replace(0, 1, inplace=True)

        df["login_gap_ratio"] = (
            df["days_since_last_login"] / df["avg_frequency_login_days"]
        )

        df["engagement_score"] = df["avg_time_spent"] * df["avg_frequency_login_days"]

        df["value_per_login"] = (
            df["avg_transaction_value"] / df["avg_frequency_login_days"]
        )

        df["wallet_utilization"] = df["points_in_wallet"] / (
            df["avg_transaction_value"] + 1
        )

        df["complaint_flag"] = (
            (df["past_complaint"] == 1) & (df["complaint_status"] != "Resolved")
        ).astype(int)

        logger.info("Feature engineering completed successfully.")
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise e

    return df


# in this class we are wrapping the pandas operationds in to a sklearn transformer, coz we need the pipeline to be fully sklearn for production grade.
class BasicPreprocessor(BaseEstimator, TransformerMixin):
    """
    A basic preprocessor that applies initial preprocessing steps.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = fill_missing_values(X)
        X = datetime_cols_conversion(X)
        X = feature_engineering(X)
        X = drop_unnecessary_columns(X)
        return X


def preprocess_df(df):
    """
    Full preprocessing pipeline for the DataFrame.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    logger.info("Starting preprocessing pipeline.")

    try:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        logger.info("Numeric pipeline created.")

        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]
        )
        logger.info("Categorical pipeline created.")

        preprocessor = Pipeline(
            [
                ("basic", BasicPreprocessor()),
                (
                    "columns",
                    ColumnTransformer(
                        transformers=[
                            (
                                "num",
                                numeric_pipeline,
                                lambda X: X.select_dtypes(
                                    include=["int64", "float64"]
                                ).columns,
                            ),
                            (
                                "cat",
                                categorical_pipeline,
                                lambda X: X.select_dtypes(
                                    include=["object", "category"]
                                ).columns,
                            ),
                        ]
                    ),
                ),
            ]
        )

        logger.info("Preprocessing pipelines created successfully.")
    except Exception as e:
        logger.error(f"Error in creating pipelines: {e}")
        raise e
    return preprocessor
