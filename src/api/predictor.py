import pandas as pd


def predict(model, input_data):
    # Wrap in list to handle single dictionary (scalars) correctly
    df = pd.DataFrame([input_data])
    churn_risk_score = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return churn_risk_score, probability
