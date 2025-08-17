from fastapi import FastAPI
from routes import predict, info

app = FastAPI(title="Loan Default Prediction API")

# Root endpoint
@app.get("/")
def root():
    return {"message": "Loan Eligibility Prediction API is running"}

# Include routers for prediction and info endpoints
app.include_router(predict.router, prefix="/predict")
app.include_router(info.router, prefix="/info")
