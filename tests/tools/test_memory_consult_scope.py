"""Tenant-isolation and org-scoping tests for the consult_memory tool.

The whole cross-org defense is one predicate — `.eq("client_id", org_id)` in
`_fetch_allowed_notes`. The forwarded `allowed_note_ids` list rides the request
and is therefore client-influenced, so if that predicate is ever dropped in a
refactor the service-role client will happily hand back another org's notes.
These tests pin the behavior down with a fake Supabase that really filters, so
removing the predicate makes them fail rather than merely changing a call log.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import memory_consult_tool as mct  # noqa: E402


# ---------------------------------------------------------------------------
# A fake Supabase that actually applies the filters, not just records them.
# ---------------------------------------------------------------------------

class _FakeQuery:
    def __init__(self, rows):
        self._rows = [dict(r) for r in rows]
        self.applied = []

    def select(self, *_args, **_kwargs):
        return self

    def in_(self, column, values):
        self.applied.append(("in", column))
        allowed = set(values)
        self._rows = [r for r in self._rows if r.get(column) in allowed]
        return self

    def eq(self, column, value):
        self.applied.append(("eq", column))
        self._rows = [r for r in self._rows if r.get(column) == value]
        return self

    def or_(self, expr):
        # PostgREST `.or_("is_user_context.eq.false,user_id.eq.<caller>")` — the
        # personal-overlay predicate. Recorded but not applied: these fixtures
        # exercise the org (client_id) boundary and don't carry the overlay
        # columns, so filtering here would drop otherwise-valid rows.
        self.applied.append(("or", expr))
        return self

    def limit(self, n):
        self._rows = self._rows[:n]
        return self

    def execute(self):
        return SimpleNamespace(data=self._rows)


class _FakeSupabase:
    def __init__(self, tables):
        self._tables = tables
        self.queries = []

    def schema(self, _name):
        return self

    def table(self, name):
        query = _FakeQuery(self._tables.get(name, []))
        self.queries.append((name, query))
        return query


HOME_ORG = "org-home"
OTHER_ORG = "org-other"
USER_ID = "user-1"

OWN_NOTE = {
    "id": "note-own",
    "path": "Team/Pricing",
    "description": "",
    "content": "Our Atlas pricing tier is $4,200/mo.",
    "internal_summary": "",
    "context_node_id": "node-1",
    "client_id": HOME_ORG,
}
FOREIGN_NOTE = {
    "id": "note-foreign",
    "path": "Team/Secrets",
    "description": "",
    "content": "COMPETITOR CONFIDENTIAL — margin is 62%.",
    "internal_summary": "",
    "context_node_id": "node-9",
    "client_id": OTHER_ORG,
}


def _users(active_org_id=None, client_id=HOME_ORG):
    return [{"id": USER_ID, "client_id": client_id, "active_org_id": active_org_id}]


@pytest.fixture(autouse=True)
def llm_spy():
    """Capture the prompt handed to the sub-agent LLM.

    autouse: consult_memory calls the sub-agent whenever it fetched any note
    content, so without a stub in EVERY test a passing assertion could still be
    firing a real OpenRouter request (or passing only because the failure was
    caught and swallowed). Tests that assert on the prompt just request it.
    """
    seen = {}

    def _call_llm(*, provider, model, messages, **kwargs):
        seen["messages"] = messages
        seen["prompt"] = "\n".join(m["content"] for m in messages)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"status": "ok", "summary": "distilled"})
                    )
                )
            ]
        )

    fake_module = SimpleNamespace(call_llm=_call_llm)
    with patch.dict(sys.modules, {"agent.auxiliary_client": fake_module}):
        yield seen


def _run(sb, allowed_note_ids, user_id=USER_ID):
    scope = {
        "allowed_note_ids": allowed_note_ids,
        "root_node_id": "node-root",
        "root_kind": "team",
        "scope_node_ids": ["node-root"],
    }
    with patch.object(mct, "_get_memory_context", return_value=scope), patch.object(
        mct, "_get_user_id", return_value=user_id
    ), patch.object(mct, "_get_supabase", return_value=sb):
        return json.loads(mct.consult_memory("what is our pricing?"))


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------

def test_forged_allow_list_cannot_reach_another_orgs_notes(llm_spy):
    """A note id from another org in the allow-list must match zero rows."""
    sb = _FakeSupabase({"users": _users(), "user_notes": [OWN_NOTE, FOREIGN_NOTE]})

    result = _run(sb, ["note-foreign"])

    # Nothing was distilled, because nothing was fetched.
    assert "prompt" not in llm_spy, "sub-agent LLM must not be called with no notes"
    assert result["status"] == "ok"
    assert "no readable content" in result["summary"]


def test_foreign_note_content_never_reaches_the_subagent_prompt(llm_spy):
    """Mixed allow-list: the caller's own note is used, the foreign one is not."""
    sb = _FakeSupabase({"users": _users(), "user_notes": [OWN_NOTE, FOREIGN_NOTE]})

    result = _run(sb, ["note-own", "note-foreign"])

    assert result["status"] == "ok"
    assert "Atlas pricing tier" in llm_spy["prompt"]
    assert "COMPETITOR CONFIDENTIAL" not in llm_spy["prompt"]


def test_note_fetch_is_constrained_by_client_id():
    """The org predicate must be present on the note query itself."""
    sb = _FakeSupabase({"users": _users(), "user_notes": [OWN_NOTE]})

    _run(sb, ["note-own"])

    notes_queries = [q for (name, q) in sb.queries if name == "user_notes"]
    assert notes_queries, "expected a user_notes query"
    assert ("eq", "client_id") in notes_queries[0].applied


def test_fails_closed_when_the_org_cannot_be_resolved(llm_spy):
    """No resolvable org → no note fetch at all, rather than an unscoped one."""
    sb = _FakeSupabase({"users": [], "user_notes": [OWN_NOTE, FOREIGN_NOTE]})

    result = _run(sb, ["note-own"])

    assert result == {"error": "Memory service unavailable"}
    assert not [q for (name, q) in sb.queries if name == "user_notes"]


# ---------------------------------------------------------------------------
# Org scoping — must agree with the web app that built the allow-list
# ---------------------------------------------------------------------------

def test_uses_the_active_org_when_the_user_has_switched(llm_spy):
    """The web app scopes the allow-list to the ACTIVE org, so this must too.

    app/api/chat/memory/route.ts resolves the acting org as
    `membership.orgId ?? user.clientId` — the same coalesce the JWT claim and
    app.current_user_org_id() use. Re-scoping on raw client_id here would match
    zero rows for any user whose active org differs from their home org, and the
    tool would silently report "no relevant memory".
    """
    active_note = dict(OWN_NOTE, id="note-active", client_id=OTHER_ORG,
                       content="Active-org note body.")
    sb = _FakeSupabase(
        {
            "users": _users(active_org_id=OTHER_ORG),
            "memberships": [
                {"user_id": USER_ID, "org_id": OTHER_ORG, "status": "active"}
            ],
            "user_notes": [OWN_NOTE, active_note],
        }
    )

    result = _run(sb, ["note-active"])

    assert result["status"] == "ok"
    assert "Active-org note body." in llm_spy["prompt"]


def test_stale_active_org_without_membership_falls_back_to_home_org(llm_spy):
    """A stale active_org_id must NOT grant access to that org's notes.

    active_org_id is a user-writable selection. Honoring it without checking for
    a live membership would let a removed member keep reading their old org — so
    an unvalidated coalesce is a security regression, not just a bug fix.
    """
    sb = _FakeSupabase(
        {
            "users": _users(active_org_id=OTHER_ORG),
            "memberships": [],  # no active membership in OTHER_ORG
            "user_notes": [OWN_NOTE, FOREIGN_NOTE],
        }
    )

    result = _run(sb, ["note-own", "note-foreign"])

    assert result["status"] == "ok"
    assert "Atlas pricing tier" in llm_spy["prompt"]
    assert "COMPETITOR CONFIDENTIAL" not in llm_spy["prompt"]


# ---------------------------------------------------------------------------
# Truncation must be visible
# ---------------------------------------------------------------------------

class _RecordingLogger:
    """Captures warnings regardless of which logger the module bound at import.

    memory_consult_tool picks the Hermes structured logger when `src.utils.logger`
    is importable and a stdlib shim otherwise — and which one it gets depends on
    whether another test has already put the Hermes `src` package on sys.path.
    Asserting through caplog is therefore order-dependent; spying on the module's
    logger attribute is not.
    """

    def __init__(self):
        self.warnings = []

    def info(self, msg, **kwargs):
        pass

    def warning(self, msg, **kwargs):
        self.warnings.append((msg, kwargs))

    def error(self, msg, **kwargs):
        pass

    def exception(self, msg, **kwargs):
        pass


def test_allow_list_over_the_cap_warns():
    """Silently trimming the allow-list makes a partial answer look complete."""
    many = [f"note-{i}" for i in range(mct._MAX_NOTES + 5)]
    sb = _FakeSupabase({"users": _users(), "user_notes": [OWN_NOTE]})
    spy = _RecordingLogger()

    with patch.object(mct, "logger", spy):
        _run(sb, many)

    matches = [(m, kw) for (m, kw) in spy.warnings if "truncated" in m.lower()]
    assert matches, f"expected a truncation warning, got: {spy.warnings}"
    _, kwargs = matches[0]
    assert kwargs["requested"] == mct._MAX_NOTES + 5
    assert kwargs["used"] == mct._MAX_NOTES


def test_corpus_over_the_char_budget_warns(llm_spy):
    """Notes dropped for exceeding the prompt budget must be reported too."""
    big = "x" * 5_000
    notes = [
        dict(OWN_NOTE, id=f"note-{i}", path=f"Team/Note{i}", content=big)
        for i in range(20)  # 100k chars of content vs a 60k budget
    ]
    sb = _FakeSupabase({"users": _users(), "user_notes": notes})
    spy = _RecordingLogger()

    with patch.object(mct, "logger", spy):
        _run(sb, [n["id"] for n in notes])

    matches = [(m, kw) for (m, kw) in spy.warnings if "truncated" in m.lower()]
    assert matches, f"expected a corpus-truncation warning, got: {spy.warnings}"
    _, kwargs = matches[0]
    assert kwargs["notes_dropped"] > 0
    assert kwargs["included"] < len(notes)
