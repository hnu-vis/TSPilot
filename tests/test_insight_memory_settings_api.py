from fastapi.testclient import TestClient

import app.routes.resources as resource_routes
from app.server import app
from core.key_insight.learning import InsightLearningScheduleStore


def test_insight_learning_schedule_http_round_trip_and_validation(tmp_path, monkeypatch):
    store = InsightLearningScheduleStore(tmp_path / "learning", default_max_wait_seconds=600)
    monkeypatch.setattr(resource_routes, "get_insight_learning_schedule_store", lambda: store)
    client = TestClient(app)

    initial = client.get("/api/v1/resources/insight-memory-learning-settings")
    assert initial.status_code == 200
    assert initial.json()["settings"]["max_wait_seconds"] == 600

    updated = client.patch(
        "/api/v1/resources/insight-memory-learning-settings",
        json={"max_wait_seconds": 90},
    )
    assert updated.status_code == 200
    assert updated.json()["settings"]["max_wait_seconds"] == 90
    assert client.get("/api/v1/resources/insight-memory-learning-settings").json()["settings"]["max_wait_seconds"] == 90

    invalid = client.patch(
        "/api/v1/resources/insight-memory-learning-settings",
        json={"max_wait_seconds": 0},
    )
    assert invalid.status_code == 422
    assert store.read().max_wait_seconds == 90
