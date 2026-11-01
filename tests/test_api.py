"""
API tests using Flask's test client.

Data-backed endpoints need the raw dataset; /predict tests need a trained
model (models/saved/revenue_model.joblib, created by `python main.py`)
and scikit-learn. Tests skip cleanly when those are unavailable.
"""

import pytest

from analysis.data_loader import DEFAULT_RAW_PATH
from models.predictor import model_available, validate_request

pytest.importorskip("flask")

needs_data = pytest.mark.skipif(not DEFAULT_RAW_PATH.exists(), reason="raw dataset not present")
needs_model = pytest.mark.skipif(not model_available(), reason="no trained model")

VALID_BUY = {
    "date": "2020-06-15", "campaign_number": "camp 1", "user_engagement": "High",
    "banner": "240 x 400", "placement": "ghi", "displays": 20000, "cost": 25.0,
}
CATEGORIES = {
    "campaign_number": ["camp 1", "camp 2", "camp 3"],
    "user_engagement": ["High", "Low", "Medium"],
    "banner": ["240 x 400"],
    "placement": ["ghi"],
}


@pytest.fixture(scope="module")
def client():
    from api import create_app
    return create_app().test_client()


# ---------------------------------------------------------------------
# Request validation (no model needed)
# ---------------------------------------------------------------------

def test_validate_request_accepts_valid_buy():
    assert validate_request(VALID_BUY, CATEGORIES) == []


@pytest.mark.parametrize("change, message", [
    ({"banner": "1 x 1"}, "Invalid banner"),
    ({"displays": -5}, "displays must be a non-negative number"),
    ({"cost": "ten"}, "cost must be a non-negative number"),
    ({"cost": True}, "cost must be a non-negative number"),
    ({"date": "15/06/2020"}, "date must be in YYYY-MM-DD format"),
    ({"date": "2019-01-01"}, "date must be on or after"),
])
def test_validate_request_rejects_bad_fields(change, message):
    errors = validate_request({**VALID_BUY, **change}, CATEGORIES)
    assert any(message in e for e in errors), errors


def test_validate_request_accepts_rows_read_from_csv(tmp_path):
    import numpy as np
    import pandas as pd
    path = tmp_path / "buys.csv"
    pd.DataFrame([VALID_BUY, {**VALID_BUY, "displays": 500}]).to_csv(path, index=False)
    items = pd.read_csv(path).to_dict(orient="records")
    assert all(validate_request(item, CATEGORIES) == [] for item in items)
    assert validate_request({**VALID_BUY, "displays": np.int64(7)}, CATEGORIES) == []


def test_validate_request_reports_missing_fields():
    errors = validate_request({"date": "2020-06-15"}, CATEGORIES)
    assert "Missing required fields" in errors[0]


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------

def test_health(client):
    body = client.get("/health").get_json()
    assert body["status"] == "healthy"


@needs_data
def test_metrics_overall_matches_pipeline(client):
    body = client.get("/metrics").get_json()
    assert body["total_records"] == 15403
    assert body["total_revenue"] == pytest.approx(276262.62, abs=0.01)


@needs_data
def test_metrics_filters_and_errors(client):
    body = client.get("/metrics?campaign_number=camp 2").get_json()
    assert body["date_range"].endswith("2020-04-30")  # camp 2 ran in April only

    r = client.get("/metrics?banner=nope")
    assert r.status_code == 400
    assert "Invalid banner" in r.get_json()["messages"][0]

    r = client.get("/metrics?start=2021-01-01")
    assert r.status_code == 404


@needs_data
def test_metrics_by_dimension(client):
    body = client.get("/metrics/campaign").get_json()
    assert body["dimension"] == "campaign_number"
    assert {g["campaign_number"] for g in body["groups"]} == {"camp 1", "camp 2", "camp 3"}

    daily = client.get("/metrics/date?start=2020-06-01").get_json()["groups"]
    assert len(daily) == 30 and daily[0]["date"] == "2020-06-01"

    assert client.get("/metrics/channel").status_code == 404


@needs_data
def test_api_info(client):
    body = client.get("/api/info").get_json()
    assert body["dataset"]["rows"] == 15403
    assert "Unknown" in body["filters"]["placement"]


@needs_data
@needs_model
def test_predict_single_and_batch(client):
    body = client.post("/predict", json=VALID_BUY).get_json()
    assert body["predicted_revenue"] > 0
    assert body["predicted_margin"] == pytest.approx(body["predicted_revenue"] - 25.0, abs=1e-3)
    assert isinstance(body["profitable"], bool)

    later = {**VALID_BUY, "date": "2020-08-01"}
    batch = client.post("/predict", json=[VALID_BUY, later]).get_json()
    assert len(batch) == 2
    assert batch[1]["warnings"] and "after the training data" in batch[1]["warnings"][0]


@needs_model
def test_predict_rejects_invalid_input(client):
    r = client.post("/predict", json={**VALID_BUY, "placement": "zzz"})
    assert r.status_code == 400
    assert "Invalid placement" in r.get_json()["messages"][0]

    assert client.post("/predict", data="not json").status_code == 400
    assert client.post("/predict", json=[]).status_code == 400
