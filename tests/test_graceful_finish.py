"""Tests for the graceful budget finish (run_agent.py).

Production forensics 2026-07-30 (main env, 153 in-depth chat turns joined from
app.usage_events.metadata.api_calls):

    loop length      turns    share
    1-15 calls          72    47.1%
    16-45 calls         29    19.0%
    46-59 calls         10     6.5%
    >=60 (cap hit)      42    27.5%

A near-void at 46-59 next to a pile-up at exactly 60-66 is the signature of a
WALL, not a natural stopping distribution: turns were not finishing at 55, they
were being killed at 60. 7.7% of those capped turns shipped a LITERALLY EMPTY
message (status=complete, no error) against ~0-1.4% elsewhere; the rest shipped
600-1700 chars of methodology prose with no widget, after ~10 minutes and
$2.29-$5.02.

Four defects produced that:
  1. the exit prompt asked the model to summarize its PROCESS, never restating
     the user's question or the output contract;
  2. the salvage call was non-streamed, so the UI went dead at the worst moment;
  3. budget pressure was computed from api_call_count/max_iterations only, so a
     parent starved by a delegate_task subagent draining the SHARED
     IterationBudget got hard-stopped having never seen a single warning;
  4. the cap fired one call BEFORE the call that would have written the answer.

The fix: exhausting a budget no longer breaks the loop. It enters a FINISHING
phase inside the same loop — tools withdrawn, the original question restated,
the output contract restated, streaming intact — so the turn ends with the
ANSWER instead of a stub.
"""

import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_agent import AIAgent, CostBudget, IterationBudget


# ---------------------------------------------------------------------------
# Fixtures / helpers (mirrors tests/test_cost_budget.py)
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


def _make_agent(tools=("web_search", "delegate_task"), **kwargs):
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tools)),
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


def _tool_turn(name="web_search"):
    return _mock_response(
        content="", finish_reason="tool_calls", tool_calls=[_mock_tool_call(name)]
    )


def _run(agent, message="hello"):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message)


def _tool_hungry_model(answer="The answer is 42."):
    """A model that calls tools forever — until it HAS no tools.

    Faithful to how a real provider behaves: it cannot emit a tool call when
    the request carries none. That is the whole point of withdrawing them
    rather than asking the model to stop.
    """
    def _respond(**kwargs):
        if kwargs.get("tools"):
            return _tool_turn()
        return _mock_response(content=answer)
    return _respond


def _run_until_finish(agent, message="hello", answer="The answer is 42."):
    agent.client.chat.completions.create.side_effect = _tool_hungry_model(answer)
    with patch("run_agent.handle_function_call", return_value="tool output"):
        return _run(agent, message)


def _calls(agent):
    return agent.client.chat.completions.create.call_args_list


# ---------------------------------------------------------------------------
# The core fix: the budget wall becomes a finishing phase
# ---------------------------------------------------------------------------


class TestGracefulFinish:
    def test_exhausted_iterations_withdraw_tools_instead_of_breaking(self):
        """The call that writes the answer happens INSIDE the loop, tools off."""
        agent = _make_agent(max_iterations=3)
        result = _run_until_finish(agent, "what is revenue by region?")

        assert result["final_response"] == "The answer is 42."
        # 3 exploration calls, then the finishing call.
        assert len(_calls(agent)) == 4
        for call in _calls(agent)[:3]:
            assert call.kwargs["tools"], "exploration calls must still have tools"
        assert _calls(agent)[3].kwargs["tools"] is None, (
            "the finishing call must have tools withdrawn, not merely be asked "
            "nicely to stop calling them"
        )

    def test_finish_instruction_asks_for_the_answer_not_the_process(self):
        """The old prompt asked the model to summarize what it had DONE.

        That is why production capped turns shipped methodology prose. The
        finishing instruction must restate the user's QUESTION and demand the
        answer.
        """
        agent = _make_agent(max_iterations=2)
        _run_until_finish(agent, "what is revenue by region?")

        injected = _calls(agent)[-1].kwargs["messages"][-1]
        assert injected["role"] == "user"
        body = injected["content"]
        assert "what is revenue by region?" in body, (
            "the finishing prompt must restate the original question"
        )
        assert "summarizing what you've found and accomplished so far" not in body
        assert "iteration" not in body.lower(), (
            "the user's answer must not be framed around our internal budget"
        )

    def test_a_delivered_finish_reports_completed(self):
        """A forced finish that produced a real answer is COMPLETE.

        Reporting completed=False sent every capped turn through hermes's
        recovery re-prompt — the 61-66 api_calls tail seen in production.
        """
        agent = _make_agent(max_iterations=3)
        result = _run_until_finish(agent)

        assert result["completed"] is True
        assert result["budget_finished"] is True
        assert result["finish_reason"] == "iterations"

    def test_an_undelivered_finish_still_reports_incomplete(self):
        """If the finishing phase produces nothing, recovery must still fire."""
        agent = _make_agent(max_iterations=2)
        agent.client.chat.completions.create.side_effect = (
            [_tool_turn(), _tool_turn()] + [_mock_response(content="")] * 8
        )
        with patch("run_agent.handle_function_call", return_value="tool output"):
            result = _run(agent)

        assert result["budget_finished"] is True
        assert result["completed"] is False

    def test_healthy_runs_are_completely_unaffected(self):
        """A turn that answers inside its budget never enters finishing."""
        agent = _make_agent(max_iterations=60)
        agent.client.chat.completions.create.return_value = _mock_response(
            content="Quick answer"
        )
        result = _run(agent)

        assert result["final_response"] == "Quick answer"
        assert result["completed"] is True
        assert result["budget_finished"] is False
        assert result["finish_reason"] is None
        assert _calls(agent)[0].kwargs["tools"], "healthy turns keep their tools"

    def test_finishing_reserve_is_bounded(self):
        """A model that never produces content can't loop forever tool-less."""
        agent = _make_agent(max_iterations=2, finish_reserve_iterations=3)
        agent.client.chat.completions.create.side_effect = (
            [_tool_turn(), _tool_turn()] + [_mock_response(content="")] * 20
        )
        with patch("run_agent.handle_function_call", return_value="tool output"):
            _run(agent)

        # 2 exploration + 3 finishing + the last-resort salvage (which retries
        # once on empty). Bounded and small — never an open loop.
        assert len(_calls(agent)) <= 7


# ---------------------------------------------------------------------------
# Defect 3: the pressure signal tracked the wrong counter
# ---------------------------------------------------------------------------


class TestSharedBudgetStarvation:
    def test_pressure_tracks_the_shared_budget_not_just_this_agent(self):
        """A parent at 2/60 whose subagent burned 55/60 IS under pressure.

        _get_budget_warning used api_call_count/max_iterations only, so this
        parent saw nothing at all and was then hard-stopped — the reported
        "running fine, then just stops".
        """
        budget = IterationBudget(60)
        agent = _make_agent(max_iterations=60, iteration_budget=budget)
        for _ in range(55):
            budget.consume()

        warning = agent._get_budget_warning(api_call_count=2)
        assert warning is not None
        assert "2/60" not in warning, "the binding constraint is the shared pool"

    def test_shared_budget_drain_finishes_gracefully(self):
        """Draining the shared pool must finish the answer, not kill the turn."""
        budget = IterationBudget(4)
        agent = _make_agent(max_iterations=60, iteration_budget=budget)
        result = _run_until_finish(agent, "how many deals closed?")

        assert result["final_response"] == "The answer is 42."
        assert result["budget_finished"] is True
        assert result["finish_reason"] == "session_budget"
        assert _calls(agent)[-1].kwargs["tools"] is None

    def test_delegate_task_is_withdrawn_under_pressure(self):
        """Near the wall, don't let the run start a fresh 300s subagent branch."""
        agent = _make_agent(max_iterations=10)
        agent._apply_budget_pressure(api_call_count=9)
        names = {t["function"]["name"] for t in (agent._active_tools() or [])}

        assert "delegate_task" not in names, (
            "delegation drains the shared pool and starts a long branch — it "
            "must be withdrawn before the budget runs out, not after"
        )
        assert "web_search" in names, "the data tools must survive"


# ---------------------------------------------------------------------------
# Defect 4: bound wall-clock, the thing users actually feel
# ---------------------------------------------------------------------------


class TestSoftDeadline:
    def test_deadline_triggers_the_same_graceful_finish(self):
        agent = _make_agent(max_iterations=100, soft_deadline_seconds=30)
        # Pretend the turn started 5 minutes ago.
        agent.client.chat.completions.create.side_effect = [
            _tool_turn(),
            _mock_response(content="Answered before the deadline bites."),
        ]
        with patch("run_agent.handle_function_call", return_value="tool output"):
            with patch.object(
                agent, "_elapsed_seconds", side_effect=[0.0] + [999.0] * 20
            ):
                result = _run(agent)

        assert result["budget_finished"] is True
        assert result["finish_reason"] == "deadline"
        assert result["final_response"] == "Answered before the deadline bites."
        assert _calls(agent)[-1].kwargs["tools"] is None

    def test_no_deadline_by_default(self):
        agent = _make_agent(max_iterations=5)
        assert agent.soft_deadline_seconds is None
        assert agent._deadline_progress() is None

    def test_clock_restarts_each_turn_by_default(self):
        """A multi-turn CLI session must not inherit the last turn's clock."""
        agent = _make_agent(max_iterations=5, soft_deadline_seconds=30)
        agent.client.chat.completions.create.return_value = _mock_response(
            content="hi"
        )
        _run(agent)
        first = agent._turn_started_monotonic
        _run(agent)
        assert agent._turn_started_monotonic != first

    def test_pinned_clock_spans_a_second_call_for_one_user_turn(self):
        """hermes runs the SAME agent twice for one turn (recovery re-prompt).

        Without the pin, a turn that just finished at its 8-minute deadline
        would get a fresh 8 — the promise silently doubles.
        """
        agent = _make_agent(max_iterations=5, soft_deadline_seconds=30)
        agent.client.chat.completions.create.return_value = _mock_response(
            content="hi"
        )
        _run(agent)
        first = agent._turn_started_monotonic
        agent.pin_turn_deadline = True
        _run(agent)
        assert agent._turn_started_monotonic == first


# ---------------------------------------------------------------------------
# Spend: reserve enough to pay for the answer
# ---------------------------------------------------------------------------


class TestCostReserve:
    def test_near_exhausted_reserves_room_for_the_finish(self):
        b = CostBudget(6.0)
        b.add(4.0)
        assert b.near_exhausted() is False
        b.add(1.2)  # 5.2 / 6.0 = 87%
        assert b.near_exhausted() is True
        assert b.exceeded is False, (
            "we must start finishing BEFORE the ceiling so the answer call is "
            "still affordable"
        )

    def test_zero_ceiling_never_trips(self):
        assert CostBudget(0.0).near_exhausted() is False
