"""usage_listener: per-LLM-call usage deltas for external metering."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_agent as ra  # noqa: E402
from run_agent import AIAgent  # noqa: E402

# ── helpers: same shapes as tests/test_run_agent.py's _mock_response /
# _mock_assistant_msg (SimpleNamespace, not MagicMock — run_conversation's
# reasoning/content handling does real string ops on these fields, which a
# freewheeling MagicMock attribute would silently violate). ──
def _make_tool_defs(*names):
    return [{"type": "function", "function": {"name": n, "description": n,
             "parameters": {"type": "object", "properties": {}}}} for n in names]

def _usage(prompt=100, completion=50):
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}

def _mock_response(content="hi", usage=None):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = SimpleNamespace(**(usage if usage is not None else _usage()))
    return resp

def _cost_result(amount):
    c = MagicMock()
    c.amount_usd, c.status, c.source = amount, "estimated", "test"
    return c

def _make_agent(**kwargs):
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(api_key="test-key-1234567890", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, **kwargs)
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.tool_delay = 0
    a.compression_enabled = False
    return a


def test_listener_receives_one_delta_per_llm_call():
    deltas = []
    agent = _make_agent(usage_listener=deltas.append)
    agent.client.chat.completions.create.return_value = _mock_response()
    with patch("run_agent.estimate_usage_cost", return_value=_cost_result(0.25)):
        agent.run_conversation("hello")
    assert len(deltas) == 1
    d = deltas[0]
    assert d["input_tokens"] == 100 and d["output_tokens"] == 50
    assert d["cost_usd"] == 0.25 and d["model"] == agent.model


def test_listener_exception_is_swallowed():
    def boom(_):
        raise RuntimeError("listener broke")
    agent = _make_agent(usage_listener=boom)
    agent.client.chat.completions.create.return_value = _mock_response()
    with patch("run_agent.estimate_usage_cost", return_value=_cost_result(0.01)):
        result = agent.run_conversation("hello")
    assert result["final_response"]  # run still completed


def test_unpriced_call_reports_none_cost():
    deltas = []
    agent = _make_agent(usage_listener=deltas.append)
    agent.client.chat.completions.create.return_value = _mock_response()
    with patch("run_agent.estimate_usage_cost", return_value=_cost_result(None)):
        agent.run_conversation("hello")
    assert deltas[0]["cost_usd"] is None


def test_delegate_child_inherits_listener():
    # _build_child_agent does `from run_agent import AIAgent` as a *local*
    # import inside the function body, so the class must be patched on the
    # run_agent module (not on tools.delegate_tool, which never imports the
    # name at module scope).
    from tools import delegate_tool as dt
    listener = MagicMock()
    parent = _make_agent(usage_listener=listener)
    with patch("run_agent.AIAgent") as MockAgent:
        try:
            dt._build_child_agent(
                task_index=0,
                goal="t",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=5,
                parent_agent=parent,
            )
        except Exception:
            pass  # child run plumbing may fail after construction; we only need the ctor call
    assert MockAgent.call_args.kwargs.get("usage_listener") is listener
