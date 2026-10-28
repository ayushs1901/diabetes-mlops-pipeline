from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.joblib"
FEATURE_NAMES = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)

saved_artifact = joblib.load(MODEL_PATH)
model = saved_artifact["model"] if isinstance(saved_artifact, dict) else saved_artifact
app = FastAPI(title="Diabetes Prediction API")


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Pregnancies: int = Field(ge=0)
    Glucose: float = Field(ge=0, allow_inf_nan=False)
    BloodPressure: float = Field(ge=0, allow_inf_nan=False)
    SkinThickness: float = Field(ge=0, allow_inf_nan=False)
    Insulin: float = Field(ge=0, allow_inf_nan=False)
    BMI: float = Field(ge=0, allow_inf_nan=False)
    DiabetesPedigreeFunction: float = Field(ge=0, allow_inf_nan=False)
    Age: int = Field(ge=0)


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "Diabetes prediction API"}


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(payload: PredictionInput) -> dict[str, int | str]:
    row = pd.DataFrame(
        [[getattr(payload, name) for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )
    prediction = int(model.predict(row)[0])
    label = "diabetic" if prediction == 1 else "not_diabetic"
    return {"prediction": prediction, "label": label}
