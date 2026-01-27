import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_COLUMNS = [
    "age",
    "gender",
    "security_no",
    "region_category",
    "membership_category",
    "joining_date",
    "joined_through_referral",
    "referral_id",
    "preferred_offer_types",
    "medium_of_operation",
    "internet_option",
    "last_visit_time",
    "days_since_last_login",
    "avg_time_spent",
    "avg_transaction_value",
    "avg_frequency_login_days",
    "points_in_wallet",
    "used_special_discount",
    "offer_application_preference",
    "past_complaint",
    "complaint_status",
    "feedback",
    "churn_risk_score",
]


def raw_data_validation(df: pd.DataFrame):
    """
    Perform initial validation on the dataframe.
    """
    report = {"Missing Column": [], "Missing Values": [], "Invalid Values": []}
    try:
        for col in RAW_DATA_COLUMNS:
            if col not in df.columns:
                report["Missing Column"].append(col)
            else:
                if df[col].isnull().any():
                    report["Missing Values"].append(
                        f"{col} has {df[col].isnull().sum()} values missing."
                    )

        if (df["age"] < 0).any():
            report["Invalid Values"].append("age has negative values.")

        if (df["days_since_last_login"] < 0).any():
            report["Invalid Values"].append(
                "days_since_last_login has negative values."
            )

        if (df["avg_time_spent"] < 0).any():
            report["Invalid Values"].append("avg_time_spent has negative values.")

        if (df["avg_transaction_value"] < 0).any():
            report["Invalid Values"].append(
                "avg_transaction_value has negative values."
            )

        try:
            pd.to_datetime(df["joining_date"], errors="raise")
        except Exception as e:
            report["Invalid Values"].append("joining_date has invalid date format.")
            raise (f"joining_date has invalid date format and Error: {e}")

        non_numeric = (
            pd.to_numeric(df["avg_frequency_login_days"], errors="coerce").isnull()
            & df["avg_frequency_login_days"].notnull()
        )

        if non_numeric.any():
            report["Invalid Values"].append(
                "avg_frequency_login_days contains non-numeric values"
            )

        logger.info("Raw data validation completed.")

        is_vaild = False
        if len(report["Missing Column"]) > 0:
            is_vaild = False
        else:
            is_vaild = True

        return is_vaild, report
    except Exception as e:
        logger.error(f"Error during raw data validation: {e}")
        raise e


def processed_data_validation(df: pd.DataFrame):
    """
    Perform validation on the processed dataframe.

    Args:
        df (pd.DataFrame): The processed dataframe to validate.

    Returns:
        bool: True if validation passes, False otherwise.
        report: logs error or invaild entries.
    """
    try:
        report = {"Missing Values": [], "Invalid Values": []}

        for col in df.columns:
            if df[col].isnull().any():
                report["Missing Values"].append(col)

            if not np.issubdtype(df[col].dtype, np.number):
                report["Invalid Values"].append(f"{col} non-numeric")

            if pd.api.types.is_numeric_dtype(df[col]) and np.isinf(df[col]).any():
                report["Invalid Values"].append(f"{col} infinite")

        valid = all(len(v) == 0 for v in report.values())
        logger.info("Processed data validation completed.")
        return valid, report
    except Exception as e:
        logger.error(f"Error during processed data validation: {e}")
        raise e
