"""Managed agent-file guidance rendering and install writes."""

from __future__ import annotations

from pathlib import Path

from archex.client_setup.io import write_config_atomic

AGENT_GUIDANCE_START = "<!-- archex:mcp-guidance start -->"
AGENT_GUIDANCE_END = "<!-- archex:mcp-guidance end -->"
AGENT_GUIDANCE_PROMPT = "\n".join(
    [
        "## Repository context via archex (MCP)",
        (
            'For architecture, ownership, dependency, or "where is X" questions, use '
            "the archex MCP tools before reading files by hand:"
        ),
        "- `scout_repo` — compact structural map",
        "- `query_repo` — ranked code context for a question",
        "- `analyze_repo` — module/package architecture",
        "- `search_symbols` / `get_symbol` — exact symbol lookup",
        (
            "Pass the repository path as `repo_url`. In harnesses with on-demand tool "
            "discovery, activate the archex tools first. Treat archex output as context "
            "selection, not proof — verify with reads/tests before editing."
        ),
    ]
)


def render_agent_guidance_block() -> str:
    return f"{AGENT_GUIDANCE_START}\n{AGENT_GUIDANCE_PROMPT}\n{AGENT_GUIDANCE_END}\n"


def append_agent_guidance(agent_file: Path) -> bool:
    """Append the archex MCP guidance block to ``agent_file`` exactly once.

    Returns True if the block was written, False if it was already present.
    The append is non-destructive: existing content is preserved and the block
    is never duplicated on re-run.
    """
    block = render_agent_guidance_block()
    if agent_file.exists():
        existing = agent_file.read_text(encoding="utf-8")
        if AGENT_GUIDANCE_START in existing:
            return False
        new_content = existing.rstrip("\n") + "\n\n" + block if existing.strip() else block
    else:
        agent_file.parent.mkdir(parents=True, exist_ok=True)
        new_content = block
    write_config_atomic(agent_file, new_content)
    return True


def render_agent_guidance_preview(agent_file: Path) -> str:
    already_present = agent_file.exists() and AGENT_GUIDANCE_START in agent_file.read_text(
        encoding="utf-8"
    )
    status = (
        "archex MCP guidance already present; no change."
        if already_present
        else "Append archex MCP guidance block (idempotent):"
    )
    return f"Agent file: {agent_file}\n{status}\n\n{render_agent_guidance_block()}"
