"""Local, single-process HTTP interface to the existing translation pipeline."""

import logging
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from .agent import Context, Decision, TranslationAgent
from .translator import MarianTranslator

LOGGER = logging.getLogger(__name__)
LOCAL_HOSTS = ("localhost", "127.0.0.1")
LOCAL_ORIGINS = [
    f"http://{host}:{port}" for host in LOCAL_HOSTS for port in (5173, 4173)
]
POST_ORIGINS = {*LOCAL_ORIGINS, *(f"http://{host}:8000" for host in LOCAL_HOSTS)}
EXPLANATIONS = {
    Decision.ACCEPT_MT: "The agent kept the MarianMT draft.",
    Decision.REFINE: "The agent selected a revised French translation.",
    Decision.CLARIFY: "More context is needed to choose a translation.",
    Decision.FLAG: "The agent withheld the translation for review.",
}
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=3000)]
Hint = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
Mode = Literal["agent", "mt_only"]
ModelStatus = Literal["idle", "loading", "ready", "error"]


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    formality: Literal["formal", "informal"] | None = None
    speaker_gender: Literal["feminine", "masculine"] | None = None
    listener_gender: Literal["feminine", "masculine"] | None = None
    meaning: Hint | None = None


class TranslationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: Text
    mode: Mode = "agent"
    context: ContextRequest | None = None
    clarification: Hint | None = None

    @model_validator(mode="after")
    def reject_context_for_mt(self):
        if self.mode == "mt_only" and (self.context is not None or self.clarification is not None):
            raise ValueError("Context and clarification require agent mode.")
        return self


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    device: Literal["cpu"] = "cpu"
    marianmt: ModelStatus
    qwen: ModelStatus


class TranslationResponse(BaseModel):
    mode: Mode
    initial_translation: str
    decision: Decision | None = None
    status: Literal["resolved", "clarification", "flagged"]
    validation: Literal["passed", "withheld", "not_run"]
    explanation: str
    final_translation: str | None = None
    question: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def consistent_outcome(self):
        if (self.status == "resolved") != bool(self.final_translation):
            raise ValueError("Only resolved results have final French.")
        if (self.status == "clarification") != bool(self.question):
            raise ValueError("Only clarification results have a question.")
        expected = {
            "resolved": {Decision.ACCEPT_MT, Decision.REFINE},
            "clarification": {Decision.CLARIFY},
            "flagged": {Decision.FLAG},
        }
        if self.mode == "agent":
            validation = "passed" if self.status == "resolved" else "withheld"
            if self.decision not in expected[self.status] or self.validation != validation:
                raise ValueError("Agent decision and validation status must match the outcome.")
        elif self.decision is not None or self.validation != "not_run" or self.status != "resolved":
            raise ValueError("MarianMT-only results omit agent decisions and validation.")
        return self


def create_reviewer():
    from .qwen import QwenReviewer

    return QwenReviewer()


class ModelRuntime:
    """Serialize CPU model loading and inference, with one instance of each model."""

    def __init__(self, translator_factory=MarianTranslator, reviewer_factory=create_reviewer):
        self._lock = Lock()
        self._translator_factory = translator_factory
        self._reviewer_factory = reviewer_factory
        self._translator = None
        self._agent = None
        self.marianmt: ModelStatus = "idle"
        self.qwen: ModelStatus = "idle"

    def _load_models(self, mode: Mode):
        if self._translator is None:
            self.marianmt = "loading"
            try:
                self._translator = self._translator_factory()
            except Exception:
                self.marianmt = "error"
                raise
            self.marianmt = "ready"
        if mode == "agent" and self._agent is None:
            self.qwen = "loading"
            try:
                self._agent = TranslationAgent(self._translator, self._reviewer_factory())
            except Exception:
                self.qwen = "error"
                raise
            self.qwen = "ready"

    def translate(self, request: TranslationRequest) -> TranslationResponse:
        with self._lock:
            try:
                self._load_models(request.mode)
            except Exception as error:
                LOGGER.exception("Could not initialize translation models")
                raise HTTPException(503, "Models could not load. Check the backend and model cache.") from error
            try:
                if request.mode == "mt_only":
                    french = self._translator.translate(request.text)
                    return TranslationResponse(
                        mode=request.mode, initial_translation=french, final_translation=french,
                        status="resolved", validation="not_run", explanation="MarianMT translated directly.",
                    )
                context = request.context.model_dump() if request.context else {}
                if request.clarification:
                    context["meaning"] = "\n".join(filter(None, [context.get("meaning"), request.clarification]))
                result = self._agent.translate(request.text, Context(**context))
            except ValueError as error:
                LOGGER.info("Translation input could not be processed")
                raise HTTPException(422, "The input could not be processed. Try a shorter sentence or context.") from error

            status = {
                Decision.CLARIFY: "clarification",
                Decision.FLAG: "flagged",
            }.get(result.decision, "resolved")
            reason = result.reason
            if reason and reason.startswith("Qwen review failed:"):
                LOGGER.warning("Qwen inference failed")
                reason = "The agent could not complete its review."
            return TranslationResponse(
                mode=request.mode, initial_translation=result.initial_translation, decision=result.decision,
                status=status, validation="passed" if status == "resolved" else "withheld",
                explanation=EXPLANATIONS[result.decision], final_translation=result.final_translation,
                question=result.question, reason=reason,
            )


def create_app(runtime: ModelRuntime | None = None) -> FastAPI:
    runtime = runtime if runtime is not None else ModelRuntime()
    service = FastAPI(title="LinguaBridge Studio API")
    service.add_middleware(TrustedHostMiddleware, allowed_hosts=LOCAL_HOSTS)
    service.add_middleware(
        CORSMiddleware, allow_origins=LOCAL_ORIGINS, allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @service.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _error: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": "Enter valid English text and context."})

    @service.get("/api/health", response_model=HealthResponse)
    def health():
        return HealthResponse(marianmt=runtime.marianmt, qwen=runtime.qwen)

    @service.post("/api/translate", response_model=TranslationResponse)
    def translate(payload: TranslationRequest, request: Request):
        origin = request.headers.get("origin")
        if origin is not None and origin not in POST_ORIGINS:
            raise HTTPException(403, "This translation service is available to the local studio.")
        try:
            return runtime.translate(payload)
        except HTTPException:
            raise
        except Exception as error:
            LOGGER.exception("Unexpected translation failure")
            raise HTTPException(500, "Translation failed. Try again or check the backend.") from error

    return service


app = create_app()
