import pandas as pd


def predict(model, input_data):
    df = pd.DataFrame([input_data])
    churn_risk_score = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return churn_risk_score, probability
