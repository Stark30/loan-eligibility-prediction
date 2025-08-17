from fastapi import APIRouter, HTTPException
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

router = APIRouter()

# Load model
MODEL_PATH = "models/best_model.pkl"
model_pipeline = joblib.load(MODEL_PATH)

@router.get("/")
def model_info():
    try:
        df = pd.read_excel("models/loan_dataset.xlsx")
        df['Income_to_LoanRatio'] = df['Income'] / df['LoanAmount']
        df['AgeGroup'] = pd.cut(df['Age'], bins=[18,30,45,60,100], labels=['18-30','31-45','46-60','60+'])
        bins = [300,579,669,739,799,850]
        labels = ['Poor','Fair','Good','Very Good','Excellent']
        df['CreditScoreCategory'] = pd.cut(df['CreditScore'], bins=bins, labels=labels)

        numeric_features = [
            'Age','Income','LoanAmount','CreditScore','MonthsEmployed',
            'NumCreditLines','InterestRate','LoanTerm','DTIRatio','Income_to_LoanRatio'
        ]
        categorical_features = [
            'Education','EmploymentType','MaritalStatus','HasMortgage','HasDependents',
            'LoanPurpose','HasCoSigner','AgeGroup','CreditScoreCategory'
        ]

        X = df[numeric_features + categorical_features]
        y = df['Default']
        _, X_val, _, y_val = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

        try:
            preprocessor = model_pipeline.named_steps['preprocessor']
            feature_names = preprocessor.get_feature_names_out()
            feature_names = [f.split("__")[-1] for f in feature_names]
        except Exception:
            feature_names = None

        model_details = []
        for name, clf in model_pipeline.named_estimators_.items():
            if name == "lr":
                continue
            classifier = clf.named_steps['classifier']
            info = {"name": name, "type": type(classifier).__name__}

            try:
                preds = clf.predict(X_val)
                probas = clf.predict_proba(X_val)[:,1] if hasattr(clf,"predict_proba") else None
                info["performance"] = {
                    "accuracy": round(accuracy_score(y_val,preds),3),
                    "roc_auc": round(roc_auc_score(y_val,probas),3) if probas is not None else None,
                    "precision": round(precision_score(y_val,preds),3),
                    "recall": round(recall_score(y_val,preds),3),
                    "f1_score": round(f1_score(y_val,preds),3)
                }
            except Exception:
                info["performance"] = {}

            if hasattr(classifier,"feature_importances_"):
                values = classifier.feature_importances_
            elif hasattr(classifier,"coef_"):
                values = classifier.coef_[0]
            else:
                values = []

            if len(values) > 0 and feature_names and len(values) == len(feature_names):
                info["feature_importances"] = {fn: round(float(v),3) for fn,v in zip(feature_names,values)}
            elif len(values) > 0:
                info["feature_importances"] = {f"feature_{i}": round(float(v),3) for i,v in enumerate(values)}

            model_details.append(info)

        return {"voting_classifier_models": model_details}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
