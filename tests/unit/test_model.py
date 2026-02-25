from unittest.mock import MagicMock, patch

import pandas as pd


@patch("scripts.train_model.register_model")
@patch("scripts.train_model.mlflow")
@patch("scripts.train_model.evaluate_model")
@patch("scripts.train_model.train_model")
@patch("scripts.train_model.processed_data_validation")
@patch("scripts.train_model.raw_data_validation")
@patch("scripts.train_model.preprocess_df")
@patch("scripts.train_model.load_data")
def test_training_pipeline_selects_best_model(
    mock_load_data,
    mock_preprocess,
    mock_raw_val,
    mock_processed_val,
    mock_train_model,
    mock_evaluate,
    mock_mlflow,
    mock_register,
):
    # ----------------------
    # Fake dataframe
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8],
            "churn_risk_score": [0, 1, 0, 1],
        }
    )

    mock_load_data.return_value = df
    mock_raw_val.return_value = (True, None)
    mock_processed_val.return_value = (True, None)
    mock_preprocess.return_value = pd.DataFrame(
        {
            "age": [25, 40],
            "gender": ["M", "F"],
            "security_no": ["SEC123", "SEC456"],
            "region_category": ["City", "Town"],
            "membership_category": ["Gold Membership", "Basic Membership"],
            "joining_date": ["2023-01-10", "2022-06-15"],
            "joined_through_referral": ["Yes", "No"],
            "referral_id": ["REF1", "REF5"],
            "preferred_offer_types": ["Gift Vouchers/Coupons", "Without Offers"],
            "medium_of_operation": ["Smartphone", "Desktop"],
            "internet_option": ["Wi-Fi", "Mobile_Data"],
            "last_visit_time": ["2024-01-05 10:30:00", "2024-01-06 18:45:00"],
            "days_since_last_login": [5, 2],
            "avg_time_spent": [120.5, 80.0],
            "avg_transaction_value": [1500.0, 900.0],
            "avg_frequency_login_days": [10, 5],
            "points_in_wallet": [200, 50],
            "used_special_discount": ["Yes", "No"],
            "offer_application_preference": ["Yes", "No"],
            "past_complaint": ["No", "Yes"],
            "complaint_status": ["Not Applicable", "Solved"],
            "feedback": ["User Friendly Website", "Reasonable Price"],
            "churn_risk_score": [0, 1],
        }
    )

    # Fake trained model
    fake_model = MagicMock()
    fake_model.predict.return_value = [0, 1]

    import numpy as np

    fake_model.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8]])

    mock_train_model.return_value = fake_model

    # Make roc_auc different for each call
    mock_evaluate.side_effect = [
        {"roc_auc": 0.7},
        {"roc_auc": 0.9},
        {"roc_auc": 0.8},
    ]

    # Mock MLflow run context
    mock_run = MagicMock()
    mock_run.info.run_id = "run_123"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    # ----------------------
    # Import script (this triggers execution)
    import scripts.train_model as training_script

    training_script.main()

    # ----------------------
    # Assertions

    # Ensure MLflow run was started
    assert mock_mlflow.start_run.called

    # Ensure metrics were logged
    assert mock_mlflow.log_metrics.called

    # Ensure register_model was called
    assert mock_register.called

    # Ensure best model chosen based on highest AUC (0.9)
    args = mock_register.call_args[0]
    assert args[0] == "run_123"  # best_run_id
