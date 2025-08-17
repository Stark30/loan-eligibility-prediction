from fastapi import APIRouter, HTTPException
import pandas as pd
import joblib
from schemas.loan_request import LoanRequest
from utils.normalization import education_map, yes_no_map, employment_map, marital_map, loan_purpose_map

router = APIRouter()

# Load model
MODEL_PATH = "models/best_model.pkl"
model_pipeline = joblib.load(MODEL_PATH)

# Prediction endpoint
@router.post("/")
def predict_risk(data: LoanRequest):
    try:
        df = pd.DataFrame([data.dict()])

        # Normalize categorical inputs
        df['Education'] = df['Education'].map(education_map).fillna("Other")
        df['EmploymentType'] = df['EmploymentType'].map(employment_map).fillna("Other")
        df['MaritalStatus'] = df['MaritalStatus'].map(marital_map).fillna("Other")
        df['HasMortgage'] = df['HasMortgage'].str.lower().map(yes_no_map).fillna("No")
        df['HasDependents'] = df['HasDependents'].str.lower().map(yes_no_map).fillna("No")
        df['HasCoSigner'] = df['HasCoSigner'].str.lower().map(yes_no_map).fillna("No")
        df['LoanPurpose'] = df['LoanPurpose'].map(loan_purpose_map).fillna("Other")

        # Derived features
        df['Income_to_LoanRatio'] = df['Income'] / df['LoanAmount']
        df['AgeGroup'] = pd.cut(
            df['Age'],
            bins=[18, 30, 45, 60, 100],
            labels=['18-30', '31-45', '46-60', '60+'],
            right=True
        )
        bins = [300, 579, 669, 739, 799, 850]
        labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        df['CreditScoreCategory'] = pd.cut(df['CreditScore'], bins=bins, labels=labels, right=True)

        # Predict using the model
        pred = model_pipeline.predict(df)
        pred_proba = model_pipeline.predict_proba(df)[:, 1]

        return {"prediction": int(pred[0]), "risk_score": float(pred_proba[0])}
    
    # Handle exceptions and return error messages
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
