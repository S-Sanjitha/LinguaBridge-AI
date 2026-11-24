from concurrent.futures import ThreadPoolExecutor
import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from linguabridge.api import ModelRuntime, TranslationResponse, create_app


@pytest.fixture
def service():
    mt = Mock()
    mt.translate.return_value = "Tu peux ouvrir la fenêtre ?"
    reviewer = Mock()
    reviewer.review.return_value = json.dumps({
        "decision": "REFINE", "text": "Vous pouvez ouvrir la fenêtre ?",
    })
    mt_factory = Mock(return_value=mt)
    reviewer_factory = Mock(return_value=reviewer)
    client = TestClient(create_app(ModelRuntime(mt_factory, reviewer_factory)), base_url="http://127.0.0.1")
    return client, mt, reviewer, mt_factory, reviewer_factory


@pytest.mark.parametrize("body", [
    {}, {"text": "  "}, {"text": 7}, {"text": "x" * 3001},
    {"text": "Hello", "mode": "agent2"},
    {"text": "Hello", "context": {"formality": "casual"}},
    {"text": "Hello", "context": {"speaker_gender": "unknown"}},
    {"text": "Hello", "context": {"listener_gender": "feminine", "extra": True}},
    {"text": "Hello", "mode": "mt_only", "context": {}},
    {"text": "Hello", "mode": "mt_only", "clarification": "detail"},
])
def test_validation_before_loading(service, body):
    client, _, _, mt_factory, reviewer_factory = service
    response = client.post("/api/translate", json=body)
    assert response.status_code == 422
    assert response.json() == {"detail": "Enter valid English text and context."}
    mt_factory.assert_not_called()
    reviewer_factory.assert_not_called()


def test_mt_only_loads_only_marian_and_reuses_it(service):
    client, mt, _, mt_factory, reviewer_factory = service
    assert client.get("/api/health").json()["marianmt"] == "idle"
    for _ in range(2):
        response = client.post("/api/translate", json={"text": " Hello ", "mode": "mt_only"})
        assert response.status_code == 200
        data = response.json()
        assert data["final_translation"] == mt.translate.return_value
        assert data["decision"] is None and data["validation"] == "not_run"
    assert client.get("/api/health").json()["qwen"] == "idle"
    mt_factory.assert_called_once()
    reviewer_factory.assert_not_called()


def test_agent_refinement_and_runtime_reuse_across_concurrent_posts(service):
    client, _, _, mt_factory, reviewer_factory = service
    body = {"text": "Can you open the window?", "context": {"formality": "formal"}}

    with ThreadPoolExecutor(max_workers=3) as pool:
        responses = list(pool.map(lambda _: client.post("/api/translate", json=body), range(3)))
    for response in responses:
        assert response.status_code == 200
        data = response.json()
        assert data["decision"] == "REFINE" and data["validation"] == "passed"
        assert data["initial_translation"] == "Tu peux ouvrir la fenêtre ?"
        assert data["final_translation"] == "Vous pouvez ouvrir la fenêtre ?"
    mt_factory.assert_called_once()
    reviewer_factory.assert_called_once()
    assert client.get("/api/health").json() == {
        "status": "ok", "device": "cpu", "marianmt": "ready", "qwen": "ready",
    }


def test_clarification_preserves_context_and_only_displays_resolved_french(service):
    client, mt, reviewer, _, _ = service
    mt.translate.return_value = "Je suis allé à la banque."
    body = {"text": "I went to the bank.", "context": {"speaker_gender": "masculine", "meaning": "At dusk."}}
    reviewer.review.return_value = json.dumps({
        "decision": "CLARIFY", "text": "Do you mean a financial institution or a riverbank?",
    })
    first = client.post("/api/translate", json=body).json()
    assert first["status"] == "clarification" and first["final_translation"] is None

    mt.translate.side_effect = ["Je suis allé à la banque.", "Je suis allé à la rive."]
    reviewer.review.return_value = '{"decision":"ACCEPT_MT","text":""}'
    second = client.post("/api/translate", json={**body, "clarification": "The edge of a river."}).json()
    assert second["status"] == "resolved" and second["final_translation"] == "Je suis allé à la rive."
    context = reviewer.review.call_args.args[2]
    assert context.speaker_gender == "masculine"
    assert context.meaning == "At dusk.\nThe edge of a river."


def test_flag_withholds_final_and_sanitizes_model_error(service):
    client, _, reviewer, _, _ = service
    reviewer.review.side_effect = RuntimeError("secret cache path")
    response = client.post("/api/translate", json={"text": "Can you open the window?"})
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "FLAG" and data["status"] == "flagged"
    assert data["validation"] == "withheld" and data["final_translation"] is None
    assert "secret" not in response.text


def test_backend_errors_do_not_show_internal_messages(service):
    client, mt, _, mt_factory, _ = service
    mt_factory.side_effect = RuntimeError("private model init detail")
    init_error = client.post("/api/translate", json={"text": "Hello"})
    assert init_error.status_code == 503 and "private" not in init_error.text
    assert client.get("/api/health").json()["marianmt"] == "error"
    mt_factory.side_effect = None
    mt.translate.side_effect = RuntimeError("private inference detail")
    inference_error = client.post("/api/translate", json={"text": "Hello"})
    assert inference_error.status_code == 500 and "private" not in inference_error.text


def test_response_rejects_flagged_final_translation():
    with pytest.raises(ValidationError):
        TranslationResponse(
            mode="agent", initial_translation="Bonjour", final_translation="Bonjour",
            decision="FLAG", status="flagged", validation="withheld", explanation="Withheld.",
        )


def test_cors_allows_only_configured_origins(service):
    client = service[0]
    for origin, expected in (("http://localhost:5173", True), ("https://example.com", False)):
        response = client.options("/api/translate", headers={
            "Origin": origin, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        })
        assert ("access-control-allow-origin" in response.headers) == expected


def test_rejects_foreign_browser_posts_and_hosts_without_initializing(service):
    client, _, _, mt_factory, _ = service
    response = client.post("/api/translate", json={"text": "Hello"}, headers={"Origin": "https://example.com"})
    assert response.status_code == 403
    response = client.get("/api/health", headers={"Host": "example.com"})
    assert response.status_code == 400
    mt_factory.assert_not_called()
