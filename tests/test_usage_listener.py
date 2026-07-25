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

def _mock_response(content="hi", usage=None, tool_calls=None, finish_reason=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    resolved_finish_reason = finish_reason or ("tool_calls" if tool_calls else "stop")
    choice = SimpleNamespace(message=msg, finish_reason=resolved_finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = SimpleNamespace(**(usage if usage is not None else _usage()))
    return resp

def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or "call_1",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )

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


def test_listener_receives_one_delta_per_llm_call_across_tool_round_trip():
    # Mirrors tests/test_run_agent.py's TestRunConversation.test_tool_calls_then_stop:
    # first response has tool_calls (loop continues), second is a final "stop".
    deltas = []
    agent = _make_agent(usage_listener=deltas.append)
    tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
    resp1 = _mock_response(content="", tool_calls=[tc], usage=_usage(100, 20))
    resp2 = _mock_response(content="Done searching", usage=_usage(150, 30))
    agent.client.chat.completions.create.side_effect = [resp1, resp2]
    with (
        patch("run_agent.handle_function_call", return_value="search result"),
        patch("run_agent.estimate_usage_cost", return_value=_cost_result(0.1)),
    ):
        result = agent.run_conversation("search something")
    assert result["final_response"] == "Done searching"
    assert len(deltas) == 2
    assert deltas[0]["input_tokens"] == 100 and deltas[0]["output_tokens"] == 20
    assert deltas[1]["input_tokens"] == 150 and deltas[1]["output_tokens"] == 30


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
    assert MockAgent.call_args is not None, (
        "AIAgent ctor never reached — _build_child_agent failed before construction"
    )
    assert MockAgent.call_args.kwargs.get("usage_listener") is listener


def test_background_review_agent_inherits_listener():
    # _spawn_background_review's _run_review constructs the review AIAgent with
    # usage_listener=self.usage_listener (run_agent.py ~1420). That kwarg *was*
    # the fix in 02104d1c — this test pins it so a refactor dropping it fails
    # loudly. The review thread is run inline (fake Thread) for determinism;
    # `AIAgent` resolves as a run_agent module global at call time, so patching
    # run_agent.AIAgent intercepts the ctor.
    import threading

    listener = MagicMock()
    parent = _make_agent(usage_listener=listener)

    class _InlineThread:
        def __init__(self, target=None, daemon=None, name=None):
            self._target = target

        def start(self):
            self._target()

    with (
        patch("run_agent.AIAgent") as MockAgent,
        patch.object(threading, "Thread", _InlineThread),
    ):
        parent._spawn_background_review([], review_memory=True)
    assert MockAgent.call_args is not None, (
        "review AIAgent ctor never reached — _run_review failed before construction"
    )
    assert MockAgent.call_args.kwargs.get("usage_listener") is listener


def test_listener_shared_across_concurrent_agents_receives_all_deltas():
    # Threading smoke test for the documented contract: one listener may be
    # fired from multiple threads (main loop, background review, delegate
    # children each run their own agent instance). Two agents sharing a
    # listener run concurrently; both deltas must land, unmangled.
    import threading

    deltas = []
    lock = threading.Lock()

    def listener(d):
        with lock:
            deltas.append(d)

    agents = [_make_agent(usage_listener=listener) for _ in range(2)]
    for i, a in enumerate(agents):
        a.client.chat.completions.create.return_value = _mock_response(
            usage=_usage(100 + i, 50)
        )

    with patch("run_agent.estimate_usage_cost", return_value=_cost_result(0.05)):
        threads = [
            threading.Thread(target=a.run_conversation, args=("hello",))
            for a in agents
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "agent thread did not finish"
    assert len(deltas) == 2
    assert sorted(d["input_tokens"] for d in deltas) == [100, 101]
    assert all(d["output_tokens"] == 50 and d["cost_usd"] == 0.05 for d in deltas)
