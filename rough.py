from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Define request model
class LoanRequest(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float
    Education: str  
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str

# Load model
MODEL_PATH = "model/best_model.pkl"
model_pipeline = joblib.load(MODEL_PATH)

# Normalization dictionaries
education_map = {
    "High School": "High School",
    "Bachelor": "Bachelor's",
    "Bachelors": "Bachelor's",
    "Master": "Master's",
    "Masters": "Master's",
    "PhD": "PhD",
    "Other": "Other"
}

yes_no_map = {"yes": "Yes", "no": "No"}

employment_map = {
    "Full-time": "Full-time",
    "Full time": "Full-time",
    "Part-time": "Part-time",
    "Part time": "Part-time",
    "Unemployed": "Unemployed",
    "Other": "Other"
}

marital_map = {
    "Single": "Single",
    "Married": "Married",
    "Divorced": "Divorced",
    "Widowed": "Widowed",
    "Other": "Other"
}

loan_purpose_map = {
    "Auto": "Auto",
    "Home": "Home",
    "Education": "Education",
    "Business": "Business",
    "Other": "Other"
}

@app.post("/predict")
def predict_risk(data: LoanRequest):
    try:
        # Convert to DataFrame
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

        # Predict
        pred = model_pipeline.predict(df)
        pred_proba = model_pipeline.predict_proba(df)[:, 1]

        return {"prediction": int(pred[0]), "risk_score": float(pred_proba[0])}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from fastapi import HTTPException
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

@app.get("/info")
def model_info():
    try:
        # Load dataset
        df = pd.read_excel("model/loan_dataset.xlsx")
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

        # Define numeric & categorical features (must match your trained model)
        numeric_features = [
            'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
            'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Income_to_LoanRatio'
        ]
        categorical_features = [
            'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 
            'LoanPurpose', 'HasCoSigner', 'AgeGroup', 'CreditScoreCategory'
        ]

        # Features & target
        X = df[numeric_features + categorical_features]
        y = df['Default']

        # Split into validation (optional)
        _, X_val, _, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Extract feature names from preprocessor
        try:
            preprocessor = model_pipeline.named_steps['preprocessor']
            feature_names = preprocessor.get_feature_names_out()
            feature_names = [f.split("__")[-1] for f in feature_names]
        except Exception:
            feature_names = None

        model_details = []

        # Iterate over estimators in VotingClassifier
        for name, clf in model_pipeline.named_estimators_.items():
            if name == "lr":  # skip logistic regression if desired
                continue

            classifier = clf.named_steps['classifier']  
            info = {
                "name": name,
                "type": type(classifier).__name__,
            }

            # Predictions & performance
            try:
                preds = clf.predict(X_val)
                probas = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else None

                info["performance"] = {
                    "accuracy": round(accuracy_score(y_val, preds), 3),
                    "roc_auc": round(roc_auc_score(y_val, probas), 3) if probas is not None else None,
                    "precision": round(precision_score(y_val, preds), 3),
                    "recall": round(recall_score(y_val, preds), 3),
                    "f1_score": round(f1_score(y_val, preds), 3)
                }
            except Exception:
                info["performance"] = {}

            # Feature importances / coefficients
            if hasattr(classifier, "feature_importances_"):
                values = classifier.feature_importances_
            elif hasattr(classifier, "coef_"):
                values = classifier.coef_[0]
            else:
                values = []

            if len(values) > 0 and feature_names and len(values) == len(feature_names):
                info["feature_importances"] = {fn: round(float(v), 3) for fn, v in zip(feature_names, values)}
            elif len(values) > 0:
                info["feature_importances"] = {f"feature_{i}": round(float(v), 3) for i, v in enumerate(values)}

            model_details.append(info)

        return {"voting_classifier_models": model_details}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
