from fastapi import FastAPI
from routes import predict, info

app = FastAPI(title="Loan Default Prediction API")

@app.get("/")
def root():
    return {"message": "Loan Eligibility Prediction API is running"}

app.include_router(predict.router, prefix="/predict")
app.include_router(info.router, prefix="/info")
