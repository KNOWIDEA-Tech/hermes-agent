"""User Notes Tool — access to the user's personal notes vault.

Provides three READ tools for the agent to discover and read user-authored
notes stored in Supabase (app.user_notes), plus ONE narrowly-scoped write tool
(`update_memory`) that can ONLY touch the auto-maintained project "AI Memory"
note (is_agent_memory=true, memory_kind='project'). The agent still CANNOT
create, update, or delete any user-authored note.

Security:
- user_id is NEVER accepted as a tool parameter
- Primary source: contextvar (coroutine-isolated, safe for concurrent requests)
- Fallback: os.environ["HERMES_USER_ID"] (for subprocess sandbox compatibility)
- Database RLS enforces filtering via session variable as defense-in-depth
- update_memory writes are additive-or-snapshotted: every edit records an
  app.agent_memory_edits provenance row so the web app can undo it.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

_SCHEMA = "app"
_TABLE = "user_notes"


def _get_user_id() -> Optional[str]:
    """Get the current user's ID from contextvar (preferred) or env (fallback).

    This is the programmatic firewall: the agent never supplies a user_id.
    It is injected server-side before the agent starts.
    
    Priority:
    1. contextvar (coroutine-isolated, safe for concurrent Modal requests)
    2. os.environ (for code running in subprocess sandbox)
    """
    # Try contextvar first (isolated per coroutine)
    try:
        from src.utils.user_context import get_user_id as get_contextvar_user_id
        user_id = get_contextvar_user_id()
        if user_id:
            return user_id
    except ImportError:
        pass
    
    # Fallback to env var (for subprocess sandbox)
    return os.environ.get("HERMES_USER_ID") or None


def _get_active_project_id() -> Optional[str]:
    """Get the selected project scope for note tools.

    Returns:
        - A UUID string: scope tools to that project only
        - Empty string "": no project selected, tools should be blocked
        - None: legacy/unscoped behavior (rare, only if env not set at all)
    """
    try:
        from src.utils.user_context import get_active_project_id

        project_id = get_active_project_id()
        if project_id is not None:
            return project_id
    except ImportError:
        pass

    return os.environ.get("HERMES_ACTIVE_PROJECT_ID")


def _exclude_detached_memory(query):
    """Hide the AI-maintained memory note from the read tools when this chat has
    memory detached (per-chat toggle OFF). Detached must mean unreachable by ANY
    path — not just absent from the injected context — otherwise the model could
    still list/search/read the note as if it were an ordinary note."""
    if _get_memory_enabled() is False:
        return query.eq("is_agent_memory", False)
    return query


def _get_supabase():
    """Return a Supabase client using service role credentials."""
    from supabase import create_client

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        return None
    return create_client(url, key)


def _set_rls_context(sb, user_id: str) -> None:
    """Set the session variable for RLS enforcement.
    
    Even though we filter with .eq("user_id", user_id), the database
    RLS policy also checks current_setting('app.current_user_id').
    This provides defense-in-depth: if app code has a bug, DB still blocks.
    
    Uses the app.set_user_context wrapper function which must be created
    in Supabase.
    """
    try:
        # Use the wrapper function (not public.set_config directly)
        sb.rpc("set_user_context", {"user_id": user_id}).execute()
    except Exception as e:
        logger.warning("[user_notes] Failed to set RLS context: %s", e)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def list_user_notes(**kwargs) -> str:
    """List all notes for the current user (paths, descriptions, timestamps)."""
    user_id = _get_user_id()
    if not user_id:
        return json.dumps({"error": "User context not available"}, ensure_ascii=False)

    sb = _get_supabase()
    if not sb:
        return json.dumps({"error": "Notes service unavailable"}, ensure_ascii=False)

    try:
        # Set RLS context for defense-in-depth
        _set_rls_context(sb, user_id)
        active_project_id = _get_active_project_id()
        
        # No project selected = no notes access (Company Context is in system prompt already)
        if active_project_id == "":
            logger.info(
                "[user_notes] list_user_notes: no project selected, returning empty for user %s",
                user_id,
            )
            return json.dumps({
                "notes": [],
                "count": 0,
                "message": "No project selected. Select a project in the chat header to access notes.",
            }, ensure_ascii=False)
        
        query = (
            sb.schema(_SCHEMA)
            .table(_TABLE)
            .select("id, path, description, project_id, created_at, updated_at")
            .eq("user_id", user_id)
        )
        query = _exclude_detached_memory(query)
        if active_project_id is not None:
            query = query.eq("project_id", active_project_id)
            logger.info(
                "[user_notes] list_user_notes: scoped to project %s for user %s",
                active_project_id,
                user_id,
            )
        result = query.order("path").order("id").execute()
        notes = result.data or []
        logger.info(
            "[user_notes] list_user_notes: returned %d notes for user %s project=%s",
            len(notes),
            user_id,
            active_project_id,
        )
        return json.dumps({
            "notes": notes,
            "count": len(notes),
        }, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error("[user_notes] list_user_notes failed for user %s: %s", user_id, e)
        return json.dumps({"error": f"Failed to list notes: {e}"}, ensure_ascii=False)


def read_user_note(path: str, note_id: Optional[str] = None, **kwargs) -> str:
    """Read a single note's full content by path (or by note_id when paths collide)."""
    user_id = _get_user_id()
    if not user_id:
        return json.dumps({"error": "User context not available"}, ensure_ascii=False)

    sb = _get_supabase()
    if not sb:
        return json.dumps({"error": "Notes service unavailable"}, ensure_ascii=False)

    nid = (note_id or "").strip() if note_id else ""

    try:
        _set_rls_context(sb, user_id)
        active_project_id = _get_active_project_id()
        
        # No project selected = no notes access
        if active_project_id == "":
            return json.dumps({
                "error": "No project selected. Select a project in the chat header to read notes.",
            }, ensure_ascii=False)

        if nid:
            query = (
                sb.schema(_SCHEMA)
                .table(_TABLE)
                .select("id, path, description, content, created_at, updated_at")
                .eq("user_id", user_id)
                .eq("id", nid)
            )
            query = _exclude_detached_memory(query)
            if active_project_id is not None:
                query = query.eq("project_id", active_project_id)
            result = query.limit(1).execute()
            notes = result.data or []
            if not notes:
                logger.info(
                    "[user_notes] read_user_note: note not found by id for user %s",
                    user_id,
                )
                return json.dumps(
                    {"error": f"Note not found: id {nid}"}, ensure_ascii=False
                )
            row = notes[0]
            p = row.get("path") or ""
            logger.info(
                "[user_notes] read_user_note: read by id for user %s path='%s' (%d bytes)",
                user_id,
                p,
                len(row.get("content") or ""),
            )
            return json.dumps({"note": row}, ensure_ascii=False, default=str)

        if not path or not path.strip():
            return json.dumps({"error": "path is required"}, ensure_ascii=False)

        query = (
            sb.schema(_SCHEMA)
            .table(_TABLE)
            .select("id, path, description, content, created_at, updated_at")
            .eq("user_id", user_id)
            .eq("path", path.strip())
        )
        query = _exclude_detached_memory(query)
        if active_project_id is not None:
            query = query.eq("project_id", active_project_id)
        result = query.execute()
        notes = result.data or []
        if len(notes) > 1:
            logger.info(
                "[user_notes] read_user_note: ambiguous path '%s' (%d matches) user %s",
                path.strip(),
                len(notes),
                user_id,
            )
            return json.dumps(
                {
                    "error": "Multiple notes use this path; pass note_id from list_user_notes.",
                    "ambiguous_path": path.strip(),
                    "matches": [
                        {"id": n.get("id"), "path": n.get("path")} for n in notes
                    ],
                },
                ensure_ascii=False,
                default=str,
            )
        if not notes:
            logger.info(
                "[user_notes] read_user_note: note not found at '%s' for user %s",
                path,
                user_id,
            )
            return json.dumps({"error": f"Note not found: {path}"}, ensure_ascii=False)

        logger.info(
            "[user_notes] read_user_note: read '%s' for user %s (%d bytes)",
            path,
            user_id,
            len(notes[0].get("content") or ""),
        )
        return json.dumps({"note": notes[0]}, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(
            "[user_notes] read_user_note failed for user %s, path '%s': %s",
            user_id,
            path,
            e,
        )
        return json.dumps({"error": f"Failed to read note: {e}"}, ensure_ascii=False)


def search_user_notes(query: str, **kwargs) -> str:
    """Full-text search across the user's notes (description + content).

    Uses Postgres tsvector/tsquery for stemmed matching:
    'running' matches 'ran', 'run', 'runs', etc.
    """
    user_id = _get_user_id()
    if not user_id:
        return json.dumps({"error": "User context not available"}, ensure_ascii=False)

    if not query or not query.strip():
        return json.dumps({"error": "query is required"}, ensure_ascii=False)

    sb = _get_supabase()
    if not sb:
        return json.dumps({"error": "Notes service unavailable"}, ensure_ascii=False)

    try:
        # Set RLS context for defense-in-depth
        _set_rls_context(sb, user_id)
        active_project_id = _get_active_project_id()
        
        # No project selected = no notes access
        if active_project_id == "":
            return json.dumps({
                "results": [],
                "count": 0,
                "query": query.strip(),
                "message": "No project selected. Select a project in the chat header to search notes.",
            }, ensure_ascii=False, default=str)
        
        # Convert query to tsquery format: split words, join with &
        words = query.strip().split()
        ts_query = " & ".join(words)

        search_query = (
            sb.schema(_SCHEMA)
            .table(_TABLE)
            .select("id, path, description, content, created_at, updated_at")
            .eq("user_id", user_id)
            .text_search("fts", ts_query, options={"type": "websearch"})
        )
        search_query = _exclude_detached_memory(search_query)
        if active_project_id is not None:
            search_query = search_query.eq("project_id", active_project_id)
        result = search_query.limit(20).execute()
        notes = result.data or []

        # Build snippets: first 200 chars of content for preview
        results = []
        for note in notes:
            results.append({
                "id": note.get("id"),
                "path": note["path"],
                "description": note.get("description"),
                "snippet": (note.get("content") or "")[:200],
                "created_at": note.get("created_at"),
                "updated_at": note.get("updated_at"),
            })

        logger.info("[user_notes] search_user_notes: query='%s' returned %d results for user %s", query.strip(), len(results), user_id)
        return json.dumps({
            "results": results,
            "count": len(results),
            "query": query.strip(),
        }, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error("[user_notes] search_user_notes failed for user %s, query='%s': %s", user_id, query, e)
        return json.dumps({"error": f"Failed to search notes: {e}"}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# update_memory — the ONLY write tool. Scoped strictly to the project's
# auto-maintained "AI Memory" note; mirrors the web app's
# lib/services/memory/agent-memory-service.ts semantics (dedup, lazy create,
# snapshot-before-replace, agent_memory_edits provenance for undo).
# ---------------------------------------------------------------------------

_MEMORY_NODE_PATH = "AI Memory"
_MEMORY_NODE_DESCRIPTION = (
    "Facts the assistant has learned about this project (auto-maintained)."
)
_EDITS_TABLE = "agent_memory_edits"
_MAX_MEMORY_BULLET_CHARS = 300
_MAX_APPEND_BULLETS = 3
_MAX_MEMORY_CHARS = 10000


def _get_message_id() -> Optional[str]:
    """Current chat turn's assistant message id (provenance/idempotency only).

    CONTEXTVAR ONLY — deliberately no os.environ fallback. This tool runs
    in-process on the agent's executor thread, where both entry points
    (_run_agent and _retry_agent in unified_chat) set the contextvar. A process
    env var is shared across concurrent requests and could stamp another turn's
    message id on this user's provenance rows.
    """
    try:
        from src.utils.user_context import get_message_id

        return get_message_id() or None
    except ImportError:
        return None


def _get_memory_enabled() -> Optional[bool]:
    """Per-chat auto-memory switch. False = writes disabled; None = unknown (on).

    CONTEXTVAR ONLY — see _get_message_id. An env fallback here could let a
    concurrent request's ON leak into a chat whose user explicitly turned
    memory OFF.
    """
    try:
        from src.utils.user_context import get_memory_enabled

        return get_memory_enabled()
    except ImportError:
        return None


def _normalize_bullet(value: str) -> str:
    """Case/punctuation-insensitive key for dedup (mirror of the TS normalize).
    A leading "the " is dropped so trivial rephrasings ("The user wants X" vs
    "User wants X") dedup to the same key."""
    import re as _re

    normalized = _re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return _re.sub(r"^the\s+", "", normalized)


def _bullets_from_content(content: Optional[str]) -> list:
    """Split note markdown back into bullet strings (mirror of the TS helper)."""
    import re as _re

    out = []
    for line in (content or "").split("\n"):
        b = _re.sub(r"^[-*]\s?", "", line).strip()
        if b:
            out.append(b)
    return out


# Placeholder outputs an LLM emits to mean "no facts" — never save as bullets
# (a "output NOTHING" prompt once produced a literal "NOTHING" memory bullet).
_NON_FACT_SENTINELS = {
    "nothing", "none", "n/a", "na", "no facts", "no new facts",
    "nothing to save", "nothing durable", "no durable facts", "empty", "null",
}


def _is_non_fact(bullet: str) -> bool:
    import re as _re

    normalized = _re.sub(r"[^a-z/ ]", "", bullet.lower())
    normalized = _re.sub(r"\s+", " ", normalized).strip()
    return normalized in _NON_FACT_SENTINELS


def _is_invalid_memory_shape(bullet: str) -> bool:
    """Memory bullets must be DECLARATIVE facts about the user — never questions,
    assistant-voice sentences, option lists, or placeholder phrases (observed
    live: an assistant clarify reply and '(empty response)' saved as memory)."""
    import re as _re

    return bool(
        _re.search(r"\?\s*$", bullet)
        or _re.match(r"^(i|i'm|i've|i'll|i'd)\b", bullet, _re.IGNORECASE)
        or _re.match(r"^(could|would|can|will|do|did) you\b", bullet, _re.IGNORECASE)
        or _re.match(r"^please\b", bullet, _re.IGNORECASE)
        or _re.match(r"^\d+[.)]\s", bullet)
        or _re.match(
            r"^\(?\s*(?:empty|no)\s+(?:response|reply|output|content|facts?)\s*\)?\.?$",
            bullet,
            _re.IGNORECASE,
        )
    )


def _bullets_from_input(content: str) -> list:
    """Coerce tool-arg content into normalized, capped bullet lines."""
    return [
        b[:_MAX_MEMORY_BULLET_CHARS]
        for b in _bullets_from_content(content)
        if not _is_non_fact(b) and not _is_invalid_memory_shape(b)
    ]


def update_memory(operation: str, content: str, **kwargs) -> str:
    """Append to / replace the project's auto-maintained AI Memory note."""
    user_id = _get_user_id()
    logger.info(
        "[Memory][tool] update_memory called: op=%s content_len=%d user=%s",
        operation,
        len(content or ""),
        user_id,
    )
    if not user_id:
        logger.warning("[Memory][tool] rejected: no user context")
        return json.dumps({"error": "User context not available"}, ensure_ascii=False)

    project_id = _get_active_project_id()
    if not project_id:
        logger.info("[Memory][tool] rejected: no project selected (user=%s)", user_id)
        return json.dumps(
            {"error": "No project selected — project memory is unavailable for this chat."},
            ensure_ascii=False,
        )

    if operation not in ("append", "replace"):
        logger.info("[Memory][tool] rejected: bad operation '%s' (user=%s)", operation, user_id)
        return json.dumps(
            {"error": "operation must be 'append' or 'replace'"}, ensure_ascii=False
        )
    new_bullets = _bullets_from_input(content or "")
    if not new_bullets:
        logger.info("[Memory][tool] rejected: empty content (user=%s)", user_id)
        return json.dumps({"error": "content must contain at least one fact"}, ensure_ascii=False)

    sb = _get_supabase()
    if not sb:
        return json.dumps({"error": "Notes service unavailable"}, ensure_ascii=False)

    # Respect the per-chat auto-memory toggle (chat header switch, threaded via
    # contextvar/env from the request payload).
    if _get_memory_enabled() is False:
        logger.info("[Memory][tool] rejected: per-chat memory toggle OFF (user=%s)", user_id)
        return json.dumps(
            {"error": "Memory is turned OFF for this chat. Do not attempt to save memory."},
            ensure_ascii=False,
        )

    try:
        _set_rls_context(sb, user_id)

        # Locate the (single) AI Memory node for this project.
        node_res = (
            sb.schema(_SCHEMA)
            .table(_TABLE)
            .select("id, path, content")
            .eq("user_id", user_id)
            .eq("project_id", project_id)
            .eq("is_agent_memory", True)
            .eq("memory_kind", "project")
            .limit(1)
            .execute()
        )
        node = (node_res.data or [None])[0]
        existing_bullets = _bullets_from_content(node.get("content") if node else None)

        if operation == "append":
            seen = {_normalize_bullet(b) for b in existing_bullets}
            accepted = []
            for b in new_bullets:
                if len(accepted) >= _MAX_APPEND_BULLETS:
                    break
                key = _normalize_bullet(b)
                if not key or key in seen:
                    continue
                accepted.append(b)
                seen.add(key)
            if not accepted:
                logger.info("[Memory][tool] no-op: all facts already in memory (user=%s)", user_id)
                return json.dumps(
                    {"result": "no_new_facts", "message": "All facts are already in memory."},
                    ensure_ascii=False,
                )
            final_bullets = existing_bullets + accepted
            added_lines = accepted
            prev_content = None
        else:  # replace
            # Shrink-guard: a full rewrite must not silently drop most of the
            # memory. Teach the model in-loop instead of accepting the loss.
            if len(existing_bullets) >= 4 and len(new_bullets) < len(existing_bullets) / 2:
                logger.info(
                    "[Memory][tool] rejected: shrink-guard (%d existing -> %d incoming, user=%s)",
                    len(existing_bullets),
                    len(new_bullets),
                    user_id,
                )
                return json.dumps(
                    {
                        "error": (
                            f"replace must contain the COMPLETE updated memory. Memory has "
                            f"{len(existing_bullets)} facts but you sent {len(new_bullets)}. "
                            "Include every retained fact, or use operation='append' to add."
                        )
                    },
                    ensure_ascii=False,
                )
            final_bullets = new_bullets
            added_lines = new_bullets
            prev_content = node.get("content") if node else None

        next_content = "\n".join(f"- {b}" for b in final_bullets)
        if len(next_content) > _MAX_MEMORY_CHARS:
            logger.info(
                "[Memory][tool] rejected: over cap (%d > %d chars, user=%s)",
                len(next_content),
                _MAX_MEMORY_CHARS,
                user_id,
            )
            return json.dumps(
                {
                    "error": (
                        f"Memory would exceed {_MAX_MEMORY_CHARS} chars. Use operation='replace' "
                        "with a consolidated, shorter set of facts."
                    )
                },
                ensure_ascii=False,
            )

        if node:
            sb.schema(_SCHEMA).table(_TABLE).update({"content": next_content}).eq(
                "id", node["id"]
            ).eq("user_id", user_id).execute()
            note_id = node["id"]
        else:
            # Lazy create on the first fact for this project.
            ins = (
                sb.schema(_SCHEMA)
                .table(_TABLE)
                .insert(
                    {
                        "user_id": user_id,
                        "project_id": project_id,
                        "path": _MEMORY_NODE_PATH,
                        "content": next_content,
                        "description": _MEMORY_NODE_DESCRIPTION,
                        "is_important": False,
                        "is_company_context": False,
                        "is_attached": False,
                        "is_agent_memory": True,
                        "memory_kind": "project",
                    }
                )
                .execute()
            )
            note_id = (ins.data or [{}])[0].get("id")
            if not note_id:
                return json.dumps({"error": "Failed to create memory note"}, ensure_ascii=False)

        # Provenance row → powers the web app's toast + undo (best-effort).
        edit_id = None
        try:
            edit_res = (
                sb.schema(_SCHEMA)
                .table(_EDITS_TABLE)
                .insert(
                    {
                        "user_id": user_id,
                        "note_id": note_id,
                        "message_id": _get_message_id(),
                        "op": operation,
                        "added_lines": added_lines,
                        "prev_content": prev_content,
                    }
                )
                .execute()
            )
            edit_id = (edit_res.data or [{}])[0].get("id")
        except Exception as edit_err:
            logger.warning("[user_notes] update_memory: provenance write failed: %s", edit_err)

        logger.info(
            "[Memory][tool] wrote: op=%s facts=%d user=%s project=%s note=%s edit=%s",
            operation,
            len(added_lines),
            user_id,
            project_id,
            note_id,
            edit_id,
        )
        return json.dumps(
            {
                "result": "ok",
                "operation": operation,
                "note_id": note_id,
                "edit_id": edit_id,
                "facts_written": len(added_lines),
                "memory_size_bullets": len(final_bullets),
            },
            ensure_ascii=False,
            default=str,
        )
    except Exception as e:
        logger.error("[user_notes] update_memory failed for user %s: %s", user_id, e)
        return json.dumps({"error": f"Failed to update memory: {e}"}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_user_notes_requirements() -> bool:
    """Check if Supabase credentials and user context are available."""
    has_supabase = bool(
        os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )
    has_user = bool(_get_user_id())
    return has_supabase and has_user


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

LIST_USER_NOTES_SCHEMA: Dict[str, Any] = {
    "name": "list_user_notes",
    "description": (
        "List all of the user's personal notes (ids, paths, project association, descriptions). "
        "Use this first to discover what notes the user has. "
        "Returns metadata only — not full content. "
        "To read a note's body, use read_user_note with path, or note_id when paths are duplicated."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

READ_USER_NOTE_SCHEMA: Dict[str, Any] = {
    "name": "read_user_note",
    "description": (
        "Read the full content of a specific user note by path, or by note_id when several notes "
        "share the same path (e.g. same title in different projects). Prefer note_id from list_user_notes "
        "when the tool reports an ambiguous path."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The note's display path or title (e.g. 'preferences/style.md')",
            },
            "note_id": {
                "type": "string",
                "description": "UUID of the note from list_user_notes — use when path is ambiguous",
            },
        },
        "required": [],
    },
}

UPDATE_MEMORY_SCHEMA: Dict[str, Any] = {
    "name": "update_memory",
    "description": (
        "Update the project's auto-maintained 'AI Memory' note — ONLY facts that would "
        "change how you answer a DIFFERENT question in a FUTURE conversation: standing "
        "preferences, explicit corrections, metric/term definitions, hard constraints, "
        "data-source facts, or explicit asks to remember. If the user repeats or insists "
        "on something, ALWAYS capture it. Do NOT save the topic of the current "
        "conversation, what the user is working on or asking about right now, their "
        "current goals/tasks, one-off request parameters, transient values, or small "
        "talk — conversation content is not memory. Most turns need NO save, and no save "
        "means DO NOT CALL THIS TOOL — never call it with placeholder content like "
        "'nothing' or 'none'; only call when you have a real fact to write. The current "
        "memory is shown in your context under 'Project Memory (AI-maintained)'. Never "
        "save a fact already in that memory (even reworded), already stated in the "
        "Company Context or the user's notes, or obvious from the project itself — "
        "memory is only for NEW information your context does not already contain. "
        "operation='append' adds new facts not already in memory; operation='replace' "
        "rewrites the WHOLE memory (content must be the complete updated set — use it to "
        "correct, dedupe, or consolidate). The user sees every update and can undo it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["append", "replace"],
                "description": "'append' = add new fact(s); 'replace' = rewrite the complete memory",
            },
            "content": {
                "type": "string",
                "description": (
                    "The fact(s), one per line as short factual sentences. For 'replace' this "
                    "must be the ENTIRE updated memory, not just the changes."
                ),
            },
        },
        "required": ["operation", "content"],
    },
}

SEARCH_USER_NOTES_SCHEMA: Dict[str, Any] = {
    "name": "search_user_notes",
    "description": (
        "Search the user's notes by keyword. Uses full-text search with stemming "
        "(e.g., 'running' matches 'ran', 'run'). Use this when the user has many "
        "notes and you need to find relevant ones without reading all of them. "
        "Returns matching note paths, descriptions, and content snippets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keywords (e.g., 'chart color preferences')",
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="list_user_notes",
    toolset="user_notes",
    schema=LIST_USER_NOTES_SCHEMA,
    handler=lambda args, **kw: list_user_notes(**kw),
    check_fn=_check_user_notes_requirements,
    emoji="📝",
)

registry.register(
    name="read_user_note",
    toolset="user_notes",
    schema=READ_USER_NOTE_SCHEMA,
    handler=lambda args, **kw: read_user_note(
        path=(args.get("path") or ""),
        note_id=args.get("note_id"),
        **kw,
    ),
    check_fn=_check_user_notes_requirements,
    emoji="📖",
)

registry.register(
    name="search_user_notes",
    toolset="user_notes",
    schema=SEARCH_USER_NOTES_SCHEMA,
    handler=lambda args, **kw: search_user_notes(
        query=args.get("query", ""),
        **kw,
    ),
    check_fn=_check_user_notes_requirements,
    emoji="🔍",
)

registry.register(
    name="update_memory",
    toolset="user_notes",
    schema=UPDATE_MEMORY_SCHEMA,
    handler=lambda args, **kw: update_memory(
        operation=args.get("operation", ""),
        content=args.get("content", ""),
        **kw,
    ),
    check_fn=_check_user_notes_requirements,
    emoji="🧠",
)
