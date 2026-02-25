from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    age: int = Field(..., example=30)
    gender: str = Field(..., example="M")
    security_no: str = Field(..., example="SEC123")
    region_category: str = Field(..., example="City")
    membership_category: str = Field(..., example="Gold Membership")
    joining_date: str = Field(..., example="2023-01-10")
    joined_through_referral: str = Field(..., example="Yes")
    referral_id: Optional[str] = Field(None, example="REF1")
    preferred_offer_types: str = Field(..., example="Without Offers")
    medium_of_operation: str = Field(..., example="Smartphone")
    internet_option: str = Field(..., example="Wi-Fi")
    last_visit_time: str = Field(..., example="2024-01-05 10:30:00")
    days_since_last_login: int = Field(..., example=5)
    avg_time_spent: float = Field(..., example=120.5)
    avg_transaction_value: float = Field(..., example=1500.0)
    avg_frequency_login_days: float = Field(..., example=10)
    points_in_wallet: float = Field(..., example=200)
    used_special_discount: str = Field(..., example="Yes")
    offer_application_preference: str = Field(..., example="Yes")
    past_complaint: str = Field(..., example="No")
    complaint_status: str = Field(..., example="Solved")
    feedback: str = Field(..., example="Good Service")


class PredictResponse(BaseModel):
    churn_risk_score: int = Field(..., example=1)
    probability: float = Field(..., example=0.82)


class BatchPredictRequest(BaseModel):
    inputs: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., example="churn_prediction_model")
    stage: str = Field(..., example="Production")
    version: int = Field(..., example=2)
    run_id: str = Field(..., example="abcd1234")
    registered_at: datetime
    roc_auc: float = Field(..., example=0.91)
