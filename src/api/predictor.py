def predict(model, df):
    churn_risk_score = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    return churn_risk_score, probability
