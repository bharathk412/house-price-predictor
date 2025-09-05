from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_price, batch_predict
from schemas import HousePredictionRequest, PredictionResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

# Health response model
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

app = FastAPI(
    title="House Price Prediction API",
    description=(
        "An API for predicting house prices based on various features. "
        "This application is part of the MLOps Bootcamp by School of Devops. "
        "Authored by Gourav Shah."
    ),
    version="1.0.0",
    contact={
        "name": "School of Devops",
        "url": "https://schoolofdevops.com",
        "email": "learn@schoolofdevops.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation
instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app)  # Creates the hidden /metrics endpoint

# Optional: Documented metrics endpoint (shows up in Swagger)
@app.get("/metrics-docs", tags=["Metrics"])
def get_metrics():
    return Response(instrumentator.registry.generate_latest(), media_type="text/plain")

# Health endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", model_loaded=True)

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: HousePredictionRequest):
    return predict_price(request)

# Batch prediction endpoint
@app.post("/batch-predict", response_model=list[PredictionResponse])
async def batch_predict_endpoint(requests: list[HousePredictionRequest]):
    return batch_predict(requests)