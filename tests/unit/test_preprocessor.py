import pandas as pd
import numpy as np
from pytest_mock import mocker
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

VAILD_VALUE_DF  = pd.DataFrame({
    'age': [25, 40],
    'gender': ['M', 'F'],
    'security_no': ['SEC123', 'SEC456'],
    'region_category': ['City', 'Town'],
    'membership_category': ['Gold Membership', 'Basic Membership'],
    'joining_date': ['2023-01-10', '2022-06-15'],
    'joined_through_referral': ['Yes', 'No'],
    'referral_id': ['REF1', 'REF5'],
    'preferred_offer_types': ['Gift Vouchers/Coupons', 'Without Offers'],
    'medium_of_operation': ['Smartphone', 'Desktop'],
    'internet_option': ['Wi-Fi', 'Mobile_Data'],
    'last_visit_time': ['2024-01-05 10:30:00', '2024-01-06 18:45:00'],
    'days_since_last_login': [5, 2],
    'avg_time_spent': [120.5, 80.0],
    'avg_transaction_value': [1500.0, 900.0],
    'avg_frequency_login_days': [10, 5],
    'points_in_wallet': [200, 50],
    'used_special_discount': ['Yes', 'No'],
    'offer_application_preference': ['Yes', 'No'],
    'past_complaint': ['No', 'Yes'],
    'complaint_status': ['Not Applicable', 'Solved'],
    'feedback': ['User Friendly Website', 'Reasonable Price'],
    'churn_risk_score': [0, 1]
    })
INVALID_VALUE_DF = pd.DataFrame({
        'age': [-5],  # invalid
        'gender': ['M'],
        'security_no': ['SEC123'],
        'region_category': ['City'],
        'membership_category': ['Gold Membership'],
        'joining_date': ['invalid-date'],  # invalid date
        'joined_through_referral': ['Yes'],
        'referral_id': ['REF1'],
        'preferred_offer_types': ['Without Offers'],
        'medium_of_operation': ['Smartphone'],
        'internet_option': ['Wi-Fi'],
        'last_visit_time': ['2024-01-05'],
        'days_since_last_login': [-2],  # invalid
        'avg_time_spent': [100],
        'avg_transaction_value': [500],
        'avg_frequency_login_days': ['abc'],  # invalid
        'points_in_wallet': [50],
        'used_special_discount': ['Yes'],
        'offer_application_preference': ['Yes'],
        'past_complaint': ['No'],
        'complaint_status': ['Solved'],
        'feedback': ['Poor Website'],
        'churn_risk_score': [1]
    })

MISSING_COLUMN_VALUE_DF = pd.DataFrame({
        'age': [25, None],
        'gender': ['M', None],
        'security_no': ['SEC123', 'SEC124'],
        'region_category': [None, None],          # all missing → Missing Values
        'membership_category': ['Gold', 'Silver'],
        'joining_date': ['2023-01-01', None],
        'joined_through_referral': ['Yes', 'No'],
        'preferred_offer_types': [None, None],    # all missing → Missing Values
        'medium_of_operation': ['Mobile', 'Web'],
        'internet_option': ['Wi-Fi', None],
        'last_visit_time': ['2024-01-01', None],
        'days_since_last_login': [5, None],
        'avg_time_spent': [120.0, None],
        'avg_transaction_value': [500.0, None],
        'avg_frequency_login_days': [3, None],
        'points_in_wallet': [100.0, None],
        'used_special_discount': ['Yes', None],
        'offer_application_preference': ['Yes', None],
        'past_complaint': ['No', None],
        'complaint_status': ['Solved', None],
        'churn_risk_score': [0, 1]
    })

PREPROCESS_SAMPLE_DF  = pd.DataFrame({
        'age': [5],  
        'gender': ['M'],
        'security_no': ['SEC123'],
        'region_category': [np.nan],
        'membership_category': ['Gold Membership'],
        'joining_date': ['2023-01-01'],  
        'joined_through_referral': ['Yes'],
        'referral_id': ['REF1'],
        'preferred_offer_types': ['Without Offers'],
        'medium_of_operation': ['Smartphone'],
        'internet_option': ['Wi-Fi'],
        'last_visit_time': ['2024-01-05'],
        'days_since_last_login': [5], 
        'avg_time_spent': [100],
        'avg_transaction_value': [500],
        'avg_frequency_login_days': ['15'],  
        'points_in_wallet': [50],
        'used_special_discount': ['Yes'],
        'offer_application_preference': ['Yes'],
        'past_complaint': ['No'],
        'complaint_status': ['Solved'],
        'feedback': ['Poor Website'],
        'churn_risk_score': [1]
    }) 
POST_VALIDATION_VALID_DF =  pd.DataFrame({
    'age': [25, 30],
    'days_since_last_login': ['5', 10],
    'avg_time_spent': [np.nan, 200],
    'avg_transaction_value': [500, 600],
    'avg_frequency_login_days': [10, 15],
    'points_in_wallet': [np.inf, 80],
    'customer_tenure_days': [300, 400],

    'gender_F': [1, 0],
    'gender_M': [0, 1],
    'gender_Unknown': [0, 0],

    'region_category_City': [np.inf, 0],
    })


from src.data.loader import load_data
def test_load_data(mocker):
    fake_df = pd.DataFrame({'age':[12,20,30],'gender':['M','F','F']})
    mocker.patch('pandas.read_csv',return_value = fake_df)

    df = load_data('fake_path.csv')
    assert df.shape == (3,2)
    assert list(df.columns) == ['age','gender']

from src.data.validator import raw_data_validation
def test_raw_data_validation_on_valid_dataset(mocker):
    report = raw_data_validation(VAILD_VALUE_DF)

    assert report['Missing Column'] == []
    assert report['Missing Values'] == []
    assert report['Invalid Values'] == []
    
def test_raw_data_validation_on_invalid_values(mocker):
    report = raw_data_validation(INVALID_VALUE_DF)

    assert report['Missing Column'] == []
    assert report['Missing Values'] == []
    assert len(report['Invalid Values']) > 0

def test_raw_data_validation_on_missing_col_and_values(mocker):
    report = raw_data_validation(MISSING_COLUMN_VALUE_DF)

    assert 'feedback' in report['Missing Column']
    assert 'referral_id' in report['Missing Column']
    assert len(report['Missing Values']) > 0
    assert len(report['Invalid Values']) == 0


from src.data.preprocessor import BasicPreprocessor
def test_basic_preprocessor(mocker):
    bp = BasicPreprocessor()
    transformed_df = bp.transform(PREPROCESS_SAMPLE_DF)
 
    assert (transformed_df['region_category'] == 'Unknown').all()
    assert 'security_no' not in transformed_df.columns
    assert 'referral_id' not in transformed_df.columns
    assert 'joining_date' not in transformed_df.columns
    assert 'last_visit_time' not in transformed_df.columns

from src.data.preprocessor import preprocess_df
def test_preprocessor_pipeline(mocker):
    preprocessor_tranformer = preprocess_df(VAILD_VALUE_DF)
    processed_data = preprocessor_tranformer.fit_transform(VAILD_VALUE_DF)
    processed_df  = pd.DataFrame(processed_data)

    assert processed_data.shape[0] == 2
    assert processed_data.shape[1] > 23
    assert not processed_df.isnull().any().any()

from src.data.validator import processed_data_validation
def test_processed_data_validation(mocker):
    preprocessor = preprocess_df(VAILD_VALUE_DF)
    processed = preprocessor.fit_transform(VAILD_VALUE_DF)

    processed_df = pd.DataFrame(processed)

    valid, report = processed_data_validation(processed_df)
    assert valid is True
    assert report['Missing Values'] == []
    assert report['Invalid Values'] == []

def test_processed_data_validation_for_invaild_df(mocker):
    valid, report = processed_data_validation(POST_VALIDATION_VALID_DF)
    assert valid is False
    assert len(report['Missing Values']) > 0
    assert len(report['Invalid Values']) > 0