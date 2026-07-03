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
from contextlib import contextmanager, nullcontext
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
        # Each extra instrumentation is best-effort: one missing package must not
        # disable the whole observability layer.
        # pydantic record="failure": success-path spans (one per validate call,
        # incl. the OpenAI SDK's internals) flooded every trace with hundreds of
        # "nullable validate_python succeeded" rows; only failed validations
        # (e.g. an LLM response that doesn't parse) are worth a span.
        for name, fn, kwargs in (
                ("httpx", getattr(logfire, "instrument_httpx", None), {}),
                ("pydantic", getattr(logfire, "instrument_pydantic", None),
                 {"record": "failure"})):
            if fn is None:
                continue
            try:
                fn(**kwargs)   # httpx -> every outbound API call
            except Exception as e:
                logger.debug("logfire.instrument_%s skipped: %s", name, e)
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


@contextmanager
def _guarded(cm):
    """Wrap a Logfire span context manager so the span's own enter/exit errors
    never escape into the agent, while still recording an exception raised by the
    wrapped body (so tool/run failures still show as errors in the trace)."""
    try:
        entered = cm.__enter__()
    except Exception:
        yield None  # span failed to start; run the body with no span
        return
    exc = None
    try:
        yield entered
    except BaseException as e:  # capture caller's error to hand to __exit__
        exc = e
    try:
        if exc is not None:
            suppress = cm.__exit__(type(exc), exc, exc.__traceback__)
        else:
            suppress = cm.__exit__(None, None, None)
    except Exception:
        suppress = False  # never let a logfire exit error mask the real flow
    if exc is not None and not suppress:
        raise exc


def span(name: str, tags: list | None = None, **attributes: Any):
    """Open a Logfire span, or a no-op context manager when disabled. Hardened so
    a Logfire failure can never break the agent (logging is best-effort).

    tags render as colored chips next to the message in the Logfire live view
    (and are queryable via ``tags @> ['…']``) — keep them low-cardinality:
    a kind ("agent"/"tool"/"report"/"llm") plus one specific ("phase:research")."""
    if ENABLED and _logfire is not None:
        try:
            cm = _logfire.span(name, _tags=tags, **attributes)
        except Exception:
            return nullcontext()
        return _guarded(cm)
    return nullcontext()


def info(message: str, tags: list | None = None, **attributes: Any) -> None:
    if ENABLED and _logfire is not None:
        try:
            _logfire.info(message, _tags=tags, **attributes)
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
        # delegate_depth > 0 marks a sub-agent run (spawned via delegate_task),
        # so child agent runs are identifiable and their cost/latency visible.
        depth = getattr(self, "_delegate_depth", 0)
        attrs["delegate_depth"] = depth
        is_sub = bool(depth and depth > 0)
        name = "subagent.run" if is_sub else "hermes.run"
        with span(name, tags=["agent", "subagent" if is_sub else "chat"], **attrs):
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
        with span(f"tool.{name}", tags=["tool", str(name)],
                  tool=str(name), args_preview=_preview(fargs)):
            return func(*args, **kwargs)
    return wrapper


def instrument_report(span_name: str):
    """Wrap an async report-pipeline impl (e.g. run_report_edit_impl) in a span.

    Report impls take (id, payload_dict). We capture input HTML size + key fields,
    and the output HTML size on return, so the rendering/edit flow is visible with
    its LLM calls nested underneath. Lazily configures (Modal per-container).
    """
    def deco(func):
        @functools.wraps(func)
        async def wrapper(idarg=None, payload=None, *a, **k):
            configure_observability()
            if not ENABLED:
                return await func(idarg, payload, *a, **k)
            attrs = {"id": str(idarg)}
            if isinstance(payload, dict):
                for f in ("edit_instruction", "pipeline_type", "user_id", "session_id"):
                    if payload.get(f):
                        attrs[f] = str(payload[f])[:200]
                html = payload.get("report_html") or payload.get("html")
                if isinstance(html, str):
                    attrs["input_html_chars"] = len(html)
            # "report.chat_v2" -> chips ["report", "chat_v2"]
            report_tags = ["report"] + span_name.split(".")[1:2]
            with span(span_name, tags=report_tags, **attrs):
                try:
                    res = await func(idarg, payload, *a, **k)
                    if isinstance(res, dict):
                        out = res.get("html") or res.get("report_html")
                        if isinstance(out, str):
                            info(f"{span_name}.result", tags=["report", "result"],
                                 output_html_chars=len(out))
                    return res
                finally:
                    flush()
        return wrapper
    return deco
