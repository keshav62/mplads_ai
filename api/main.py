from fastapi import FastAPI
import pandas as pd


app = FastAPI(
    title="MPLADS AI Risk Monitoring API",
    description="AI-powered project risk and anomaly detection system",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "MPLADS AI Risk Monitoring API is running successfully"
    }