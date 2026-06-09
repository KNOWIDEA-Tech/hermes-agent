"""User Notes Tool — read-only access to the user's personal notes vault.

Provides three tools for the agent to discover and read user-authored notes
stored in Supabase (app.user_notes). The agent CANNOT create, update, or
delete notes — they are strictly read-only context.

Security:
- user_id is NEVER accepted as a tool parameter
- Primary source: contextvar (coroutine-isolated, safe for concurrent requests)
- Fallback: os.environ["HERMES_USER_ID"] (for subprocess sandbox compatibility)
- Database RLS enforces filtering via session variable as defense-in-depth
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
