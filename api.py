from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from predict import predict

app = FastAPI(
    title="Chemical Equipment Risk API",
    description="Educational API for estimating equipment failure risk from synthetic data.",
    version="1.0.0",
)

class EquipmentReading(BaseModel):
    temperature_c: float = Field(..., ge=0, le=250)
    pressure_bar: float = Field(..., ge=0, le=100)
    vibration_mm_s: float = Field(..., ge=0, le=100)
    operating_hours: float = Field(..., ge=0, le=200000)
    days_since_maintenance: float = Field(..., ge=0, le=5000)
    load_percent: float = Field(..., ge=0, le=200)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_equipment(reading: EquipmentReading):
    try:
        return predict(reading.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
