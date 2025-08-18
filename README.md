# Loan Eligibility & Risk Scoring System

This project predicts the likelihood of a loan default using ML models and provides model insights via a FastAPI backend.

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Stark30/loan-eligibility-prediction.git
cd loan-eligibility-prediction
````

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the FastAPI server:

```bash
uvicorn main:app --reload
```

5. Access the API docs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Data

### Dataset: `loan_dataset.xlsx`
Before analyzing features, the distribution of the target variable `Default`, is analysed where in this case it is very imbalanced.  
<img width="593" height="470" alt="DistributionOfDefault_1" src="https://github.com/user-attachments/assets/f40bb025-11a0-48e2-b3bb-bb50e2d7444d" />

### Features:
  * Numeric: Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, Income\_to\_LoanRatio
  * Categorical: Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner, AgeGroup, CreditScoreCategory
### Target: `Default` (0 = No default, 1 = Default)

---

## Model

### Ensemble
**VotingClassifier ensemble** combining multiple models to leverage their strengths:

- **RandomForestClassifier**  
  - An ensemble of decision trees that reduces overfitting and captures non-linear relationships.  
  - Handles both numerical and categorical features (after encoding).  
  - Provides feature importance scores to understand which inputs influence predictions most.

- **XGBoostClassifier**  
  - Gradient boosting model that builds trees sequentially to correct previous errors.  
  - Often improves performance for imbalanced datasets or complex patterns.  
  - Efficient and provides feature importance insights.

> *Note:* LogisticRegression was tried but is skipped since it is less interpretable for tree-based features.

### Preprocessing
Before feeding data to the models, we normalize and encode features:

- **StandardScaler** for numeric features:  
  - Scales numerical columns like `Income` and `LoanAmount` to have zero mean and unit variance.  
  - Helps models converge faster and improves performance.

- **OneHotEncoder** for categorical features:  
  - Converts categorical inputs like `Education` and `EmploymentType` into binary vectors.  
  - Ensures models can process non-numeric data effectively.

### Feature Engineering
Derived additional features that capture important relationships and interactions between borrower characteristics and loan details:

- **Income_to_LoanRatio** = Income ÷ LoanAmount  
  - Represents a borrower's ability to repay the loan relative to their income.

- **LoanIncomeRatio** = LoanAmount ÷ Income  
  - Captures the proportion of the loan amount compared to the borrower's income.
  <img width="547" height="389" alt="LoanIncomeRatiovsDefault" src="https://github.com/user-attachments/assets/dab8340d-6bec-40a3-aa18-662d1d0759c4" />

- **LoanTermYears** = LoanTerm ÷ 12  
  - Converts loan term from months to years to make it more interpretable and allow models to capture long-term repayment risk.

- **AgeGroup**  
  - Categorizes borrowers into age ranges (18–25, 26–35, 36–50, 51–65, 65+) to capture age-related risk patterns.
  <img width="554" height="384" alt="AgeVSDefault_2" src="https://github.com/user-attachments/assets/347151ca-d261-48bf-9581-6bd2660c4fce" />

- **CreditScoreCategory**  
  - Converts raw credit scores into categories (Poor, Fair, Good, Very Good, Excellent) to highlight creditworthiness.

- **LoanAmtBin**  
  - Bins the loan amount into quartiles (Low, Medium, High, Very High) to handle skewed distributions and model non-linear effects.

- **Interaction Features**  
  - `LoanTerm_Employment` = combination of loan term and employment type  
  - `LoanAmtBin_Employment` = combination of loan amount bin and employment type  
  - `LoanTerm_LoanAmtBin` = combination of loan term and loan amount bin  
  - These interaction features help the model capture complex patterns where the risk depends on multiple factors together, e.g., long-term loans for part-time workers may carry higher risk.

### Model Storage
- The trained model is serialized as `backend/models/best_model.pkl` using `joblib`.  
- Because the model is large (>100MB), we use **Git LFS** to store it efficiently in the repository.  
- This model is loaded by the FastAPI backend to make predictions and provide feature insights.

---

## API Usage

### 1. Predict Loan Default Risk

* **Endpoint:** `POST /predict`
* **Request JSON Example:**

```json
{
  "Age": 35,
  "Income": 50000,
  "LoanAmount": 20000,
  "CreditScore": 700,
  "MonthsEmployed": 24,
  "NumCreditLines": 3,
  "InterestRate": 12.5,
  "LoanTerm": 36,
  "DTIRatio": 0.3,
  "Education": "Bachelor's",
  "EmploymentType": "Full-time",
  "MaritalStatus": "Single",
  "HasMortgage": "No",
  "HasDependents": "No",
  "LoanPurpose": "Auto",
  "HasCoSigner": "No"
}
```

* **Response Example:**

```json
{
  "prediction": 0,
  "risk_score": 0.63
}
```

### 2. Get Model Info

* **Endpoint:** `GET /info`
* Returns performance metrics and feature importances for each model in the ensemble:

  * Accuracy
  * ROC-AUC
  * Precision, Recall, F1-Score
  * Feature importances

---

## Retraining / Regenerating Model

1. Update `loan_dataset.xlsx` or preprocessing steps in notebooks.
2. Run the training notebook `model/main.ipynb`.
3. Save the updated model as `backend/models/best_model.pkl`.
4. Push changes to GitHub.

---

## Results

Evaluated four models for predicting loan default risk: Logistic Regression, Random Forest, XGBoost, and a Combinational (Ensemble) model. Each model was assessed using ROC curves, precision-recall metrics, confusion matrices, and risk segmentation.

## Logistic Regression
The model predicts non-defaulters (class 0) very well, but struggles to correctly identify defaulters (class 1).  

<img width="569" height="469" alt="LogReg" src="https://github.com/user-attachments/assets/eada95c4-d8aa-41b3-bb19-898e21a01bed" />

## Random Forest
Random Forest improves detection slightly over Logistic Regression in separating defaulters vs. non-defaulters, but the recall for defaulters is still very low due to class imbalance. This indicates we may need to handle imbalance (e.g., with oversampling, undersampling, or class weights) to improve defaulter detection.

<img width="498" height="394" alt="RandomF" src="https://github.com/user-attachments/assets/4b601da7-ccf3-49c1-8fd6-19712795a908" />

## XGBoost
XGBoost is better for identifying high-risk loans (defaulters) due to higher recall, even though overall accuracy is lower. This aligns with our objective of predicting potential defaults effectively.

<img width="469" height="391" alt="XGBoost" src="https://github.com/user-attachments/assets/6785d4c9-185a-4aa5-9668-78d673ce6732" />

## Combinational Model
The Combinational model aggregates the predictions of Random Forest and XGBoost to leverage their complementary strengths. This ensemble approach enhances overall stability and performance, providing more robust risk predictions and reducing reliance on a single model.

<img width="496" height="399" alt="Combination" src="https://github.com/user-attachments/assets/c727f25b-dda7-4da7-9db8-95238b4a3162" />

---

## Risk Score Interpretation

The model provides a **risk score** between 0 and 1 alongside the prediction. This score represents the **probability that a borrower will default** on their loan:

- **0–0.3:** Low risk — borrower is unlikely to default.
- **0.3–0.6:** Medium risk — borrower has moderate risk; consider additional checks.
- **0.6–1.0:** High risk — borrower is likely to default; lending should be carefully considered or declined.

> The risk score allows financial institutions to **prioritize loan approvals**, adjust interest rates, or apply stricter credit checks for higher-risk applicants.

---

## Conclusion

This system demonstrates how a combination of **feature engineering, ensemble modeling, and interpretability tools** can help predict loan default risk effectively.  
The FastAPI backend makes it easy to **integrate this model into production**, allowing real-time predictions and feature insights.  
