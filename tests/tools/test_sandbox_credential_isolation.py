"""Model-run code and model-driven shells never inherit platform credentials.

Covers the code-execution child environment (build_child_env) and the terminal
subprocess sanitizer (_sanitize_subprocess_env). Both run code the model wrote,
so both must withhold anything that looks like a credential unless it was
explicitly passed through, and identity must come from the per-request sandbox
env rather than the process-wide environment shared by concurrent requests.
"""

from __future__ import annotations

import os

import pytest

from tools.code_execution_tool import build_child_env, set_sandbox_env
from tools.environments.local import _sanitize_subprocess_env

PLATFORM_SECRETS = {
    "SUPABASE_SERVICE_ROLE_KEY": "service-role",
    "SUPABASE_JWT_SECRET": "jwt-secret",
    "SNOWFLAKE_PROGRAMMATIC_ACCESS_TOKEN": "warehouse-token",
    "SNOWFLAKE_PRIVATE_KEY": "pem",
    "HERMES_EXECUTION_CONTEXT_SECRET": "hmac",
    "HERMES_INTERNAL_SECRET": "internal",
    "DEMO_REPORT_SECRET": "demo",
    "DATABASE_PASSWORD": "pw",
    "STRIPE_API_KEY": "sk",
}


@pytest.fixture
def platform_env(monkeypatch):
    for name, value in PLATFORM_SECRETS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT_IDENTIFIER", "acct")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("HERMES_USER_ID", "user-from-another-request")
    yield
    set_sandbox_env({})


def test_child_env_withholds_every_platform_secret(platform_env):
    env = build_child_env("/tmp/rpc.sock")
    leaked = sorted(set(PLATFORM_SECRETS) & set(env))
    assert not leaked, f"credentials reached model-run code: {leaked}"


def test_child_env_keeps_non_secret_connection_basics(platform_env):
    env = build_child_env("/tmp/rpc.sock")
    assert env["SNOWFLAKE_ACCOUNT_IDENTIFIER"] == "acct"
    assert env["SUPABASE_URL"] == "https://example.supabase.co"
    assert env["HERMES_RPC_SOCKET"] == "/tmp/rpc.sock"


def test_child_identity_comes_only_from_the_per_request_sandbox_env(platform_env):
    assert "HERMES_USER_ID" not in build_child_env("/tmp/rpc.sock")

    set_sandbox_env({"HERMES_USER_ID": "this-request", "HERMES_TMP_DIR": "/tmp/exec-1"})
    env = build_child_env("/tmp/rpc.sock")
    assert env["HERMES_USER_ID"] == "this-request"
    assert "/tmp/exec-1" in env["PYTHONPATH"].split(os.pathsep)


def test_host_can_pass_a_scoped_value_explicitly(platform_env):
    set_sandbox_env({"HERMES_SQL_BROKER_TOKEN": "per-execution-capability"})
    assert build_child_env("/tmp/rpc.sock")["HERMES_SQL_BROKER_TOKEN"] == "per-execution-capability"


def test_terminal_env_withholds_secret_shaped_names():
    base = {**PLATFORM_SECRETS, "PATH": "/usr/bin", "VERIPIE_WORKSPACE": "/w", "HOME": "/root"}
    env = _sanitize_subprocess_env(base)
    assert not (set(PLATFORM_SECRETS) & set(env))
    assert env["PATH"] == "/usr/bin"
    assert env["VERIPIE_WORKSPACE"] == "/w"


def test_terminal_env_keeps_explicit_extra_values_and_forced_overrides():
    env = _sanitize_subprocess_env(
        {"PATH": "/usr/bin"},
        {"TOOL_SESSION_TOKEN": "explicit", "_HERMES_FORCE_OPENAI_API_KEY": "forced"},
    )
    assert env["TOOL_SESSION_TOKEN"] == "explicit"
    assert env["OPENAI_API_KEY"] == "forced"


def test_registered_passthrough_still_reaches_the_shell(monkeypatch):
    monkeypatch.setattr("tools.env_passthrough.is_env_passthrough", lambda name: name == "EXA_API_KEY")
    env = _sanitize_subprocess_env({"EXA_API_KEY": "exa", "SUPABASE_SERVICE_ROLE_KEY": "x"})
    assert env == {"EXA_API_KEY": "exa"}
