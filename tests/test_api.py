from fastapi.testclient import TestClient

from risk.api import app, state


def test_health_ready_and_scoring():
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200

        good = client.get("/distributors/DIST-001/risk")
        assert good.status_code == 200
        body = good.json()
        assert body["high_risk"] is False
        assert body["no_history"] is False

        missing = client.get("/distributors/DIST-119/risk")
        assert missing.status_code == 200
        assert missing.json()["high_risk"] is None
        assert missing.json()["no_history"] is True

        unknown = client.get("/distributors/DIST-999/risk")
        assert unknown.status_code == 404

        portfolio = client.get("/portfolio/summary")
        assert portfolio.status_code == 200
        summary = portfolio.json()
        assert summary["distributors"] == 120
        assert summary["no_history"] == 2
        assert summary["high_risk"] + summary["low_risk"] == summary["scored"]


def test_feedback_validation():
    with TestClient(app) as client:
        saved = client.post("/feedback", json={"distributor_id": "DIST-001", "decision": "approved"})
        assert saved.status_code == 201

        missing_decision = client.post("/feedback", json={"distributor_id": "DIST-001"})
        assert missing_decision.status_code == 422

        unknown = client.post("/feedback", json={"distributor_id": "NOPE", "decision": "rejected"})
        assert unknown.status_code == 404


def test_missing_model_returns_503_and_health_stays_up():
    with TestClient(app) as client:
        state["model"] = None
        state["error"] = "missing model.joblib"
        response = client.get("/distributors/DIST-001/risk")
        assert response.status_code == 503
        assert client.get("/health").status_code == 200


def test_unexpected_error_is_json():
    with TestClient(app, raise_server_exceptions=False) as client:
        state["model"] = object()
        response = client.get("/distributors/DIST-001/risk")
        assert response.status_code == 500
        assert response.json() == {"detail": "Something went wrong"}
        assert "traceback" not in response.text.lower()
