"""
Lightweight observability layer (Pydantic Logfire) for Hermes.

Design goals (mirrors the proven aucctus-ai pattern):
  * Opt-in & safe   -- enabled ONLY when LOGFIRE_TOKEN is set; any failure during
                       setup leaves Hermes running exactly as before.
  * Decoupled       -- the rest of the codebase calls obs.span()/obs.info()/the
                       decorators here, never `logfire` directly. When Logfire is
                       absent these are no-ops with ~zero overhead.
  * Right altitude  -- decorators add spans only at meaningful boundaries
                       (an agent run, a tool call). LLM calls are captured
                       automatically by logfire.instrument_openai().

Hermes talks to OpenRouter through the *raw OpenAI SDK*, so the correct hook is
instrument_openai() -- NOT instrument_anthropic()/instrument_openai_agents() as
used elsewhere. (Verified in verification/RESULTS.md.)
"""
from __future__ import annotations

import functools
import json
import logging
import os
from contextlib import nullcontext
from typing import Any

logger = logging.getLogger(__name__)

ENABLED: bool = False
_CONFIGURED: bool = False
_logfire = None  # bound to the logfire module once configured


def configure_observability(service_name: str = "hermes-agent",
                            environment: str | None = None) -> bool:
    """Configure Logfire once per process. Returns True if observability is live.

    No-ops (returns False) when LOGFIRE_TOKEN is unset or anything goes wrong --
    Hermes must never fail to start because of logging.
    """
    global ENABLED, _CONFIGURED, _logfire
    if _CONFIGURED:
        return ENABLED
    _CONFIGURED = True  # set first: a failure below must not cause repeated retries

    if not os.getenv("LOGFIRE_TOKEN"):
        logger.debug("LOGFIRE_TOKEN unset -- Logfire observability disabled (no-op).")
        return False

    try:
        import logfire
        logfire.configure(
            service_name=service_name,
            environment=environment or os.getenv("HERMES_ENV", "dev"),
            send_to_logfire="if-token-present",
            console=False,            # Hermes prints its own console output
            # scrubbing left at default = ON (redacts api_key/password/token).
        )
        logfire.instrument_openai()    # <-- the correct hook for Hermes' OpenRouter path
        try:
            logfire.instrument_pydantic()   # capture request-model validation errors
        except Exception:
            pass  # pydantic instrumentation is a bonus, not required
        _logfire = logfire
        ENABLED = True
        logger.info("Logfire observability enabled (service=%s).", service_name)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Logfire setup failed -- continuing without it: %s", e)
        ENABLED = False
    return ENABLED


def flush() -> None:
    """Force-export buffered spans. Call before a Modal container freezes/returns."""
    if ENABLED and _logfire is not None:
        try:
            _logfire.force_flush()
        except Exception:
            pass


def capture_context():
    """Snapshot the current OpenTelemetry context so a span opened on the main
    thread can be re-attached inside a worker thread (Hermes runs LLM/tool calls
    in threads, and OTEL context is thread-local). Returns None when disabled."""
    if not ENABLED:
        return None
    try:
        from opentelemetry import context as _ctx
        return _ctx.get_current()
    except Exception:
        return None


def attach_context(captured):
    """Re-attach a captured context in the current (worker) thread. Returns a
    detach token to pass to detach_context(), or None."""
    if captured is None:
        return None
    try:
        from opentelemetry import context as _ctx
        return _ctx.attach(captured)
    except Exception:
        return None


def detach_context(token) -> None:
    if token is None:
        return
    try:
        from opentelemetry import context as _ctx
        _ctx.detach(token)
    except Exception:
        pass


def span(name: str, **attributes: Any):
    """Open a Logfire span, or a no-op context manager when disabled."""
    if ENABLED and _logfire is not None:
        return _logfire.span(name, **attributes)
    return nullcontext()


def info(message: str, **attributes: Any) -> None:
    if ENABLED and _logfire is not None:
        try:
            _logfire.info(message, **attributes)
        except Exception:
            pass


def error(message: str, **attributes: Any) -> None:
    if ENABLED and _logfire is not None:
        try:
            _logfire.error(message, **attributes)
        except Exception:
            pass


def _preview(value: Any, limit: int = 300) -> str:
    """Compact, length-capped string of tool args for span attributes."""
    try:
        s = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:
        s = str(value)
    return s if len(s) <= limit else s[:limit] + "…"


# ---- decorators (one clean line at each instrumentation boundary) -----------

def instrument_run(func):
    """Wrap an agent run method (e.g. run_conversation) in a 'hermes.run' span."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not ENABLED:
            return func(self, *args, **kwargs)
        attrs = {}
        model = getattr(self, "model", None)
        if model:
            attrs["model"] = model
        provider = getattr(self, "provider", None)
        if provider:
            attrs["provider"] = provider
        with span("hermes.run", **attrs):
            try:
                return func(self, *args, **kwargs)
            finally:
                flush()  # ensure spans export even if the process is about to stop
    return wrapper


def instrument_tool(func):
    """Wrap the tool dispatcher (handle_function_call) in a 'tool.<name>' span."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not ENABLED:
            return func(*args, **kwargs)
        name = kwargs.get("function_name") or (args[0] if args else "unknown")
        fargs = kwargs.get("function_args") or (args[1] if len(args) > 1 else None)
        with span(f"tool.{name}", tool=str(name), args_preview=_preview(fargs)):
            return func(*args, **kwargs)
    return wrapper
