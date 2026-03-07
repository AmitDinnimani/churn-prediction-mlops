import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger

DROP_COLS = ["security_no", "referral_id", "joining_date", "last_visit_time"]

TARGET = "churn_risk_score"

logger = get_logger(__name__)

# Features created/transformed by BasicPreprocessor
NUMERIC_FEATURES = [
    "age",
    "days_since_last_login",
    "avg_time_spent",
    "avg_transaction_value",
    "avg_frequency_login_days",
    "points_in_wallet",
    "customer_tenure_days",
    "last_visit_hour",
    "last_visit_dayofweek",
    "login_gap_ratio",
    "engagement_score",
    "value_per_login",
    "wallet_utilization",
    "complaint_flag",
]

CATEGORICAL_FEATURES = [
    "gender",
    "region_category",
    "membership_category",
    "joined_through_referral",
    "preferred_offer_types",
    "medium_of_operation",
    "internet_option",
    "used_special_discount",
    "offer_application_preference",
    "past_complaint",
    "complaint_status",
    "feedback",
]


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
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Parsing dates in .* format when dayfirst=False"
        )
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
            (df["past_complaint"].isin([1, "Yes"]))
            & (df["complaint_status"] != "Resolved")
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

    def set_output(self, transform=None):
        """
        Support for scikit-learn's set_output API.
        """
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = fill_missing_values(X)
        X = datetime_cols_conversion(X)
        X = feature_engineering(X)
        X = drop_unnecessary_columns(X)
        return X


def get_preprocessor():
    """
    Returns the unfitted preprocessing pipeline.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,  # Don't prefix with num__ or cat__
    )

    # Note: We return just the transformer steps that need to be in the main pipeline
    # The 'basic' step and 'columns' step will be part of the final flat pipeline.
    return column_transformer


def preprocess_df(df):
    """
    Full preprocessing pipeline for the DataFrame.
    """
    logger.info("Starting preprocessing pipeline.")

    try:
        preprocessor = get_preprocessor()
        processed_df = preprocessor.fit_transform(df)
        logger.info("Preprocessing completed successfully.")
        return processed_df
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise e
