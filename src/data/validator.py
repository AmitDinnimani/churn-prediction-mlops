import pandas as pd
import numpy as np

logger = pd.get_logger(__name__)

RAW_DATA_COLUMNS = ['age', 'gender', 'security_no', 'region_category','membership_category', 'joining_date', 'joined_through_referral','referral_id', 'preferred_offer_types', 'medium_of_operation','internet_option', 'last_visit_time', 'days_since_last_login','avg_time_spent', 'avg_transaction_value', 'avg_frequency_login_days','points_in_wallet', 'used_special_discount','offer_application_preference', 'past_complaint', 'complaint_status','feedback', 'churn_risk_score']

PROCCESSED_DATA_COLUMNS = ['age', 'days_since_last_login', 'avg_time_spent', 'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet', 'customer_tenure_days', 'login_gap_ratio', 'engagement_score', 'value_per_login', 'wallet_utilization', 'gender_F', 'gender_M', 'gender_Unknown', 'region_category_City', 'region_category_Town', 'region_category_Unknown', 'region_category_Village', 'membership_category_Basic Membership', 'membership_category_Gold Membership', 'membership_category_No Membership', 'membership_category_Platinum Membership', 'membership_category_Premium Membership', 'membership_category_Silver Membership', 'joined_through_referral_?', 'joined_through_referral_No', 'joined_through_referral_Yes', 'preferred_offer_types_Credit/Debit Card Offers', 'preferred_offer_types_Gift Vouchers/Coupons', 'preferred_offer_types_Without Offers', 'medium_of_operation_?', 'medium_of_operation_Both', 'medium_of_operation_Desktop', 'medium_of_operation_Smartphone', 'internet_option_Fiber_Optic', 'internet_option_Mobile_Data', 'internet_option_Wi-Fi', 'used_special_discount_No', 'used_special_discount_Yes', 'offer_application_preference_No', 'offer_application_preference_Yes', 'past_complaint_No', 'past_complaint_Yes', 'complaint_status_No Information Available', 'complaint_status_Not Applicable', 'complaint_status_Solved', 'complaint_status_Solved in Follow-up', 'complaint_status_Unsolved', 'feedback_No reason specified', 'feedback_Poor Customer Service', 'feedback_Poor Product Quality', 'feedback_Poor Website', 'feedback_Products always in Stock', 'feedback_Quality Customer Care', 'feedback_Reasonable Price', 'feedback_Too many ads', 'feedback_User Friendly Website', 'churn_risk_score']



def raw_data_validation(df: pd.DataFrame):
    """
    Perform initial validation on the dataframe.
    
    Args:
        df (pd.DataFrame): The dataframe to validate.
        
    Returns:
        bool: True if validation passes, False otherwise.
    """
    report = {'Missing Column': [], 'Missing Values': [], 'Invalid Values': []}
    try:
        for col in RAW_DATA_COLUMNS:
            if col not in df.columns:
                report['Missing Column'].append(col)

        for col in RAW_DATA_COLUMNS:
            if df[col].isnull().all():
                report['Missing Values'].append(f'{col} has {df[col].isnull().sum()} values missing.')

        if (df['age'] < 0).any():
            report['Invalid Values'].append('age has negative values.')
        
        if(df['days_since_last_login'] < 0).any():
            report['Invalid Values'].append('days_since_last_login has negative values.')

        if (df['avg_time_spent'] < 0).any():
            report['Invalid Values'].append('avg_time_spent has negative values.')

        if(df['avg_transaction_value'] < 0).any():
            report['Invalid Values'].append('avg_transaction_value has negative values.')

        try:
            pd.to_datetime(df['joining_date'],errors='raise')
        except Exception as e:
            report['Invalid Values'].append('joining_date has invalid date format.')

        non_numeric = pd.to_numeric(
            df['avg_frequency_login_days'], errors='coerce'
        ).isnull() & df['avg_frequency_login_days'].notnull()

        if non_numeric.any():
            report["Invalid Values"].append(
                "avg_frequency_login_days contains non-numeric values"
            )

        logger.info("Raw data validation completed.")
        return report
    except Exception as e:
        logger.error(f'Error during raw data validation: {e}')
        raise e



def processed_data_validation(df: pd.DataFrame):
    """
    Perform validation on the processed dataframe.
    
    Args:
        df (pd.DataFrame): The processed dataframe to validate.
        
    Returns:
        bool: True if validation passes, False otherwise.
    """
    report = {'Missing Column': [], 'Missing Values': [], 'Invalid Values': []}
    try:
        for col in PROCCESSED_DATA_COLUMNS:
            if col not in df.columns:
                report['Missing Column'].append(col)

            if df[col].isnull().all():
                report['Missing Values'].append(f'{col} has {df[col].isnull().sum()} values missing.')

            if not np.issubdtype(df[col].dtype, np.number):
                report['Invalid Values'].append(f'{col} has non-numeric values.')
        
            if np.isinf(df[col]).any():
                report['Invalid Values'].append(f'{col} has infinite values.')

            if (df[col].nunique() <= 1) and col != 'churn_risk_score':
                report['Invalid Values'].append(f'{col} has constant values.')
        
        valid = all(len(v) == 0 for v in report.values())
        logger.info("Processed data validation completed.")
        return valid,report
    except Exception as e:
        logger.error(f'Error during processed data validation: {e}')
        raise e