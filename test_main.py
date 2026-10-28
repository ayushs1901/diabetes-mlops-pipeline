import pytest
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)
VALID_PAYLOAD = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 79,
    "BMI": 32.0,
    "DiabetesPedigreeFunction": 0.47,
    "Age": 33,
}


def test_home() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Diabetes prediction API"}


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict() -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("diabetic", "not_diabetic")


@pytest.mark.parametrize(
    "feature",
    (
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ),
)
def test_predict_rejects_negative_feature(feature: str) -> None:
    invalid_payload = {**VALID_PAYLOAD, feature: -1}

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_predict_rejects_missing_feature() -> None:
    invalid_payload = {key: value for key, value in VALID_PAYLOAD.items() if key != "Glucose"}

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
