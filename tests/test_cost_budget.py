"""Tests for the CostBudget spend ceiling (run_agent.py).

The 2026-07-24 cost incident: every retry/iteration cap in the agent loop is
bounded individually, but their product had no global dollar bound — one chat
turn burned ~$35 producing nothing. CostBudget is that bound. These tests
cover the budget class itself and its integration with run_conversation,
using the same mocked-client pattern as tests/test_run_agent.py (no network).
"""

import sys
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_agent import AIAgent, CostBudget, IterationBudget


# ---------------------------------------------------------------------------
# Fixtures / helpers (mirrors tests/test_run_agent.py)
# ---------------------------------------------------------------------------


def _make_tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _make_agent(**kwargs):
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            **kwargs,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = False
        a.save_trajectories = False
        return a


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None, usage=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = SimpleNamespace(**usage) if usage else None
    return resp


_USAGE = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}


def _cost_result(amount):
    return SimpleNamespace(amount_usd=amount, status="estimated", source="catalog")


_RUN_PATCHES = ("_persist_session", "_save_trajectory", "_cleanup_task_resources")


def _run(agent, message="hello"):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message)


# ---------------------------------------------------------------------------
# CostBudget class semantics
# ---------------------------------------------------------------------------


class TestCostBudgetClass:
    def test_starts_empty_and_under(self):
        b = CostBudget(5.0)
        assert b.spent_usd == 0.0
        assert b.exceeded is False

    def test_add_accumulates(self):
        b = CostBudget(5.0)
        b.add(1.25)
        b.add(0.75)
        assert b.spent_usd == pytest.approx(2.0)
        assert b.exceeded is False

    def test_exceeded_at_exact_ceiling(self):
        b = CostBudget(5.0)
        b.add(5.0)
        assert b.exceeded is True

    def test_negative_and_invalid_amounts_ignored(self):
        b = CostBudget(5.0)
        b.add(-1.0)
        b.add(0.0)
        b.add("not-a-number")
        b.add(None)
        assert b.spent_usd == 0.0

    def test_thread_safety_smoke(self):
        b = CostBudget(1_000_000.0)

        def worker():
            for _ in range(1000):
                b.add(0.001)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert b.spent_usd == pytest.approx(8.0, rel=1e-6)


# ---------------------------------------------------------------------------
# run_conversation integration
# ---------------------------------------------------------------------------


class TestCostBudgetLoop:
    def test_no_budget_preserves_behavior(self):
        """Default (no cost_budget) — result carries cost_limited=False."""
        agent = _make_agent()
        agent.client.chat.completions.create.return_value = _mock_response(
            content="Final answer", finish_reason="stop"
        )
        result = _run(agent)
        assert result["final_response"] == "Final answer"
        assert result["completed"] is True
        assert result["cost_limited"] is False

    def test_under_budget_run_completes_normally(self):
        budget = CostBudget(5.0)
        agent = _make_agent(cost_budget=budget)
        agent.client.chat.completions.create.return_value = _mock_response(
            content="Final answer", finish_reason="stop", usage=_USAGE
        )
        with patch("run_agent.estimate_usage_cost", return_value=_cost_result(0.10)):
            result = _run(agent)
        assert result["final_response"] == "Final answer"
        assert result["cost_limited"] is False
        assert budget.spent_usd == pytest.approx(0.10)

    def test_mid_run_breach_stops_tools_and_summarizes(self):
        """First call prices over the ceiling → loop stops, summary answer ships."""
        budget = CostBudget(5.0)
        agent = _make_agent(cost_budget=budget)
        tc = _mock_tool_call(call_id="c1")
        resp_tools = _mock_response(
            content="", finish_reason="tool_calls", tool_calls=[tc], usage=_USAGE
        )
        resp_summary = _mock_response(content="Here is what I found so far.", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [resp_tools, resp_summary]
        with (
            patch("run_agent.handle_function_call", return_value="tool output"),
            patch("run_agent.estimate_usage_cost", return_value=_cost_result(6.0)),
        ):
            result = _run(agent, "expensive question")
        assert result["cost_limited"] is True
        assert result["final_response"] == "Here is what I found so far."
        # Exactly one full iteration + the summary call — no further tool loops.
        assert result["api_calls"] == 1
        assert agent.client.chat.completions.create.call_count == 2
        # The summary request tells the model the SPEND limit was hit.
        summary_kwargs = agent.client.chat.completions.create.call_args_list[1].kwargs
        injected = summary_kwargs["messages"][-1]
        assert injected["role"] == "user"
        assert "spend limit" in injected["content"]

    def test_pre_exhausted_budget_spends_nothing(self):
        """A run starting over-budget (e.g. recovery re-prompt) makes ZERO API calls."""
        budget = CostBudget(5.0)
        budget.add(10.0)
        agent = _make_agent(cost_budget=budget)
        result = _run(agent)
        assert result["cost_limited"] is True
        assert result["api_calls"] == 0
        agent.client.chat.completions.create.assert_not_called()
        assert "spend" in (result["final_response"] or "").lower()

    def test_unpriced_calls_do_not_trip_budget(self):
        """estimate_usage_cost returning None must not accumulate or trip."""
        budget = CostBudget(5.0)
        agent = _make_agent(cost_budget=budget)
        agent.client.chat.completions.create.return_value = _mock_response(
            content="Final answer", finish_reason="stop", usage=_USAGE
        )
        with patch("run_agent.estimate_usage_cost", return_value=_cost_result(None)):
            result = _run(agent)
        assert result["final_response"] == "Final answer"
        assert result["cost_limited"] is False
        assert budget.spent_usd == 0.0


class TestCostLimitedSummaryPath:
    """Review findings F3/F4 (PR #10): the salvage summary's copy and metering."""

    def test_summary_failure_fallback_mentions_spend_limit(self):
        """If the cost-path salvage call raises, the fallback must say SPEND
        limit — not the misleading 'maximum iterations' copy."""
        budget = CostBudget(5.0)
        agent = _make_agent(cost_budget=budget)
        tc = _mock_tool_call(call_id="c1")
        resp_tools = _mock_response(
            content="", finish_reason="tool_calls", tool_calls=[tc], usage=_USAGE
        )
        agent.client.chat.completions.create.side_effect = [
            resp_tools,
            RuntimeError("summary upstream down"),
        ]
        with (
            patch("run_agent.handle_function_call", return_value="tool output"),
            patch("run_agent.estimate_usage_cost", return_value=_cost_result(6.0)),
        ):
            result = _run(agent, "expensive question")
        assert result["cost_limited"] is True
        assert "spend limit" in result["final_response"]
        assert "maximum iterations" not in result["final_response"]

    def test_summary_call_usage_is_metered(self):
        """The salvage call's own tokens/cost must land in the session counters
        and the shared CostBudget (review finding: it was invisible spend)."""
        budget = CostBudget(5.0)
        agent = _make_agent(cost_budget=budget)
        tc = _mock_tool_call(call_id="c1")
        resp_tools = _mock_response(
            content="", finish_reason="tool_calls", tool_calls=[tc], usage=_USAGE
        )
        resp_summary = _mock_response(
            content="Summary of work so far.", finish_reason="stop", usage=_USAGE
        )
        agent.client.chat.completions.create.side_effect = [resp_tools, resp_summary]
        with (
            patch("run_agent.handle_function_call", return_value="tool output"),
            patch("run_agent.estimate_usage_cost", return_value=_cost_result(6.0)),
        ):
            result = _run(agent, "expensive question")
        assert result["cost_limited"] is True
        # Both the loop call AND the summary call are counted.
        assert agent.session_api_calls == 2
        assert budget.spent_usd == pytest.approx(12.0)
        assert result["estimated_cost_usd"] == pytest.approx(12.0)


# Field-for-field contract of the per-call delta fired by the usage_listener
# (feat/usage-listener PR); _meter_summary_usage must emit the same shape.
_LISTENER_PAYLOAD_KEYS = {
    "input_tokens", "output_tokens", "total_tokens", "reasoning_tokens",
    "cache_read_tokens", "cache_write_tokens",
    "cost_usd", "cost_status", "cost_source", "model", "provider",
}


class TestSummaryUsageListener:
    """Cross-PR fix (#10 x #11): _meter_summary_usage must fire the per-call
    usage_listener so the salvage call's spend reaches the delta stream
    (hermes's UsageFlusher), not just the session counters / CostBudget.
    Now that #11 is merged in, the main-loop call ALSO fires the listener,
    so a cost-limited run emits two deltas: the breaching main-loop call
    and the salvage summary call."""

    def _drive_salvage(self, agent):
        tc = _mock_tool_call(call_id="c1")
        resp_tools = _mock_response(
            content="", finish_reason="tool_calls", tool_calls=[tc], usage=_USAGE
        )
        resp_summary = _mock_response(
            content="Summary of work so far.", finish_reason="stop", usage=_USAGE
        )
        agent.client.chat.completions.create.side_effect = [resp_tools, resp_summary]
        with (
            patch("run_agent.handle_function_call", return_value="tool output"),
            patch("run_agent.estimate_usage_cost", return_value=_cost_result(6.0)),
        ):
            return _run(agent, "expensive question")

    def test_summary_metering_fires_listener_with_contract_payload(self):
        budget = CostBudget(5.0)
        agent = _make_agent(cost_budget=budget)
        listener = MagicMock()
        agent.usage_listener = listener
        result = self._drive_salvage(agent)
        assert result["cost_limited"] is True
        # Two deltas: the breaching main-loop call (#11's block) and the
        # salvage summary call (#10's _meter_summary_usage) — no spend path
        # is invisible to the delta stream.
        assert listener.call_count == 2
        for call in listener.call_args_list:
            delta = call.args[0]
            assert set(delta.keys()) == _LISTENER_PAYLOAD_KEYS
            assert delta["cost_usd"] == pytest.approx(6.0)
            assert delta["cost_status"] == "estimated"
            assert delta["cost_source"] == "catalog"
            assert delta["model"] == agent.model

    def test_raising_listener_does_not_break_salvage(self):
        budget = CostBudget(5.0)
        agent = _make_agent(cost_budget=budget)
        agent.usage_listener = MagicMock(side_effect=RuntimeError("listener boom"))
        result = self._drive_salvage(agent)
        assert result["cost_limited"] is True
        assert result["final_response"] == "Summary of work so far."
        # Metering itself still completed despite the raising listener.
        assert budget.spent_usd == pytest.approx(12.0)
        assert agent.session_api_calls == 2


class TestIterationBudgetOwnership:
    """Review finding F1 (PR #10): run_conversation's unconditional budget
    reset silently handed delegate children a fresh full budget, so the
    documented shared cap never constrained them."""

    def test_external_budget_is_not_replaced_by_run_conversation(self):
        shared = IterationBudget(50)
        agent = _make_agent(iteration_budget=shared)
        agent.client.chat.completions.create.return_value = _mock_response(
            content="Done", finish_reason="stop"
        )
        result = _run(agent)
        assert result["final_response"] == "Done"
        assert agent.iteration_budget is shared, (
            "externally provided (shared) budget must survive run_conversation"
        )
        assert shared.used == 1  # the child's call drained the SHARED pool

    def test_self_owned_budget_still_resets_per_turn(self):
        """CLI/gateway parents keep the existing per-turn refresh."""
        agent = _make_agent()
        agent.client.chat.completions.create.return_value = _mock_response(
            content="Done", finish_reason="stop"
        )
        _run(agent)
        first = agent.iteration_budget
        _run(agent)
        assert agent.iteration_budget is not first
        assert agent.iteration_budget.used == 1  # only the second turn's call

    def test_exhausted_shared_budget_skips_the_tool_loop(self):
        shared = IterationBudget(1)
        shared.consume()  # parent already spent the whole pool
        agent = _make_agent(iteration_budget=shared)
        agent.client.chat.completions.create.return_value = _mock_response(
            content="Done", finish_reason="stop"
        )
        result = _run(agent)
        # The tool loop never runs (remaining == 0). The ONE call made is the
        # graceful finish: tools withdrawn, and asked for the ANSWER rather
        # than a recap of a process that never happened. (A parent starved by
        # its own subagents still has all their findings in context, so this
        # call is what turns them into an answer instead of a dead turn.)
        assert agent.client.chat.completions.create.call_count == 1
        only_call = agent.client.chat.completions.create.call_args
        assert only_call.kwargs["tools"] is None
        assert "Answer the user's original question NOW" in only_call.kwargs["messages"][-1]["content"]
        assert result["api_calls"] == 0
        assert result["finish_reason"] == "session_budget"
        assert shared.used == 1  # unchanged — nothing consumed by this run


# ---------------------------------------------------------------------------
# Subagent inheritance (delegate_tool)
# ---------------------------------------------------------------------------


class TestDelegateInheritance:
    def test_child_agent_shares_parent_cost_budget(self):
        from tools.delegate_tool import _build_child_agent

        budget = CostBudget(5.0)
        parent = _make_agent(cost_budget=budget)
        # _build_child_agent does `from run_agent import AIAgent` locally,
        # so the patch target is the run_agent module attribute.
        with patch("run_agent.AIAgent") as MockAgent:
            _build_child_agent(
                task_index=0,
                goal="do a thing",
                context=None,
                toolsets=["web_tools"],
                model=None,
                max_iterations=5,
                parent_agent=parent,
            )
        assert MockAgent.call_args.kwargs["cost_budget"] is budget
