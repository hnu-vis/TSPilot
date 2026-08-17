from fastapi.testclient import TestClient

import app.routes.resources as resource_routes
import app.model_config as model_config_module
import app.deps as deps
from app.model_config import ModelConfigStore
from app.server import app
from app.settings import Settings
from core.timeseries.anomaly_registry import default_anomaly_detector_name
from core.timeseries.forecast_registry import default_forecast_model_name
from core.timeseries.forecast_registry import get_forecast_model
from core.timeseries.anomaly_registry import get_anomaly_detector, set_default_anomaly_detector, unregister_anomaly_detector


def _store(tmp_path) -> ModelConfigStore:
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="environment-llm-secret",
        OPENAI_API_BASE="https://llm.example.test/v1",
        OPENAI_MODEL="reasoning-model",
        EMBEDDING_API_KEY="environment-embedding-secret",
        EMBEDDING_API_BASE="https://embedding.example.test/v1",
        EMBEDDING_MODEL="vector-model",
        TSPILOT_MODEL_CONFIG_PATH=str(tmp_path / "models.json"),
    )
    return ModelConfigStore(settings.resolved_model_config_path, settings)


def test_model_config_http_round_trip_masks_secrets_and_updates_runtime_defaults(tmp_path, monkeypatch):
    store = _store(tmp_path)
    monkeypatch.setattr(resource_routes, "get_model_config_store", lambda: store)
    client = TestClient(app)

    initial = client.get("/api/v1/resources/models/config")
    assert initial.status_code == 200
    assert initial.json()["ai"]["llm"]["active_id"] == "llm-default"
    assert initial.json()["ai"]["llm"]["models"] == [{
        "id": "llm-default",
        "provider": "OpenAI compatible",
        "api_base": "https://llm.example.test/v1",
        "model": "reasoning-model",
        "api_key_configured": True,
        "is_active": True,
        "source": "environment",
        "config_path": None,
    }]
    assert "environment-llm-secret" not in initial.text

    updated = client.patch(
        "/api/v1/resources/models/ai/llm",
        json={
            "api_base": "https://gateway.example.test/v1/",
            "model": "new-reasoning-model",
            "api_key": "new-secret",
        },
    )
    assert updated.status_code == 200
    saved_id = updated.json()["saved_id"]
    saved = next(item for item in updated.json()["ai"]["llm"]["models"] if item["id"] == saved_id)
    assert saved["api_base"] == "https://gateway.example.test/v1"
    assert saved["model"] == "new-reasoning-model"
    llm_config_path = tmp_path / "models" / "ai" / "llm" / "new-reasoning-model.json"
    assert llm_config_path.is_file()
    assert "new-secret" not in updated.text
    assert store.effective_ai().model == "reasoning-model"

    activated = client.patch(f"/api/v1/resources/models/ai/llm/{saved_id}/activate")
    assert activated.status_code == 200
    assert activated.json()["ai"]["llm"]["active_id"] == saved_id
    assert store.effective_ai().api_key == "new-secret"
    assert store.effective_ai().model == "new-reasoning-model"

    another = client.patch(
        "/api/v1/resources/models/ai/llm",
        json={"api_base": "https://second.example.test/v1", "model": "second-model"},
    )
    assert another.status_code == 200
    assert len(another.json()["ai"]["llm"]["models"]) == 3

    removed = client.delete(f"/api/v1/resources/models/ai/llm/{another.json()['saved_id']}")
    assert removed.status_code == 200
    assert len(removed.json()["ai"]["llm"]["models"]) == 2

    selected = client.patch(
        "/api/v1/resources/models/machine-learning",
        json={
            "forecast_model": default_forecast_model_name(),
            "anomaly_detector": default_anomaly_detector_name(),
        },
    )
    assert selected.status_code == 200
    assert selected.json()["machine_learning"]["forecast_options"]
    assert selected.json()["machine_learning"]["anomaly_options"]


def test_model_config_rejects_invalid_endpoint_and_unregistered_models(tmp_path, monkeypatch):
    store = _store(tmp_path)
    monkeypatch.setattr(resource_routes, "get_model_config_store", lambda: store)
    client = TestClient(app)

    invalid_endpoint = client.patch(
        "/api/v1/resources/models/ai/embedding",
        json={"api_base": "not-a-url", "model": "vector-model"},
    )
    assert invalid_endpoint.status_code == 422

    invalid_models = client.patch(
        "/api/v1/resources/models/machine-learning",
        json={"forecast_model": "missing", "anomaly_detector": "missing"},
    )
    assert invalid_models.status_code == 422


def test_external_machine_model_uses_one_config_file_and_runtime_registry(tmp_path, monkeypatch):
    store = _store(tmp_path)
    monkeypatch.setattr(resource_routes, "get_model_config_store", lambda: store)
    client = TestClient(app)

    saved = client.patch(
        "/api/v1/resources/models/machine-learning/external/forecast",
        json={
            "name": "external-forecast-v2",
            "endpoint": "https://models.example.test/forecast",
            "api_key": "external-secret",
            "timeout_seconds": 12,
        },
    )
    assert saved.status_code == 200
    config_path = tmp_path / "models" / "machine_learning" / "forecast" / "external-forecast-v2.json"
    assert config_path.is_file()
    assert "external-secret" in config_path.read_text(encoding="utf-8")
    assert "external-secret" not in saved.text
    public = next(item for item in saved.json()["machine_learning"]["forecast_models"] if item["name"] == "external-forecast-v2")
    assert public["source"] == "api"
    assert public["config_path"] == str(config_path)
    assert get_forecast_model("external-forecast-v2").endpoint == "https://models.example.test/forecast"

    activated = client.patch("/api/v1/resources/models/machine-learning/forecast/external-forecast-v2/activate")
    assert activated.status_code == 200
    assert activated.json()["machine_learning"]["forecast_model"] == "external-forecast-v2"

    active_delete = client.delete("/api/v1/resources/models/machine-learning/external/forecast/external-forecast-v2")
    assert active_delete.status_code == 409

    assert client.patch("/api/v1/resources/models/machine-learning/forecast/linear_regression/activate").status_code == 200
    removed = client.delete("/api/v1/resources/models/machine-learning/external/forecast/external-forecast-v2")
    assert removed.status_code == 200
    assert not config_path.exists()


def test_conversation_model_selection_builds_an_isolated_llm_from_the_selected_file(tmp_path, monkeypatch):
    store = _store(tmp_path)
    connection_id = store.upsert_ai(
        "llm",
        {
            "model": "conversation-specific-model",
            "api_base": "https://conversation.example.test/v1",
            "api_key": "conversation-secret",
        },
    )
    monkeypatch.setattr(deps, "get_model_config_store", lambda: store)
    monkeypatch.setattr(deps, "_create_chat_llm", lambda settings, structured: (settings.model, settings.api_base, structured))

    service = deps.get_plain_chat_service_for_model(connection_id)

    assert service._llm == ("conversation-specific-model", "https://conversation.example.test/v1", True)


def test_external_machine_model_is_registered_from_its_file_on_startup(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.upsert_external_machine_model(
        "anomaly",
        {
            "name": "startup-detector",
            "endpoint": "https://models.example.test/anomaly",
            "api_key": "startup-secret",
            "timeout_seconds": 9,
        },
    )
    store.update_machine_learning({"anomaly_detector": "startup-detector"})
    monkeypatch.setattr(model_config_module, "get_model_config_store", lambda: store)

    try:
        model_config_module.apply_persisted_machine_learning_defaults()
        detector = get_anomaly_detector("startup-detector")
        assert detector.endpoint == "https://models.example.test/anomaly"
        assert detector.timeout_seconds == 9
        assert detector.headers == {"Authorization": "Bearer startup-secret"}
    finally:
        set_default_anomaly_detector("zscore")
        unregister_anomaly_detector("startup-detector")
