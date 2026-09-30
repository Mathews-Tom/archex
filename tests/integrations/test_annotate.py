"""Tests for `archex annotate`: search hits in, per-unit fact lines out.

Every tool-output fixture below is synthetic text shaped exactly like a format
observed in local session transcripts or recorded hook payloads: omp `grep`
single-file, header-tree, and bare forms and `glob` header tree; shell
`path:line:` output of `rg -n`, `grep -rn`, and `git grep -n`; one-file
`line:text` output; `find`/`rg --files` path lists; Claude Code `Grep`
content and file lists and `Glob` file lists (from `tool_response`); Codex
`exec_command` output; OpenCode `grep`/`glob` output (from its v1.14.33 tool
source) and `bash` output with its trailing metadata block.

`python_simple` indexes to these units (1-based, inclusive):

    services/auth.py  AuthService class L11-24, AuthService.__init__ L12-13,
                      AuthService.login L15-18, AuthService.logout L20-21,
                      AuthService.verify_token L23-24; lines 1-10 module-level
    utils.py          hash_password L9-10, validate_email L13-14
    main.py           run L8-15; lines 17-19 are covered by no chunk
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from archex.annotate import (
    MAX_CALL_TOKENS,
    MAX_LINE_TOKENS,
    ShellSearch,
    annotate,
    classify_call,
    parse_search_result,
    parse_shell_search,
)
from archex.cli.main import cli
from archex.project import init_project
from archex.reporting import count_tokens


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _index(repo: Path) -> None:
    init_project(repo)
    result = CliRunner().invoke(cli, ["index", str(repo)])
    assert result.exit_code == 0, result.output


@pytest.fixture
def indexed_repo(python_simple_repo: Path) -> Path:
    _index(python_simple_repo)
    return python_simple_repo


@pytest.fixture
def diagnostics_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log_path = tmp_path / "hook-diagnostics.log"
    monkeypatch.setenv("ARCHEX_HOOK_DIAGNOSTICS_LOG", str(log_path))
    return log_path


def _unit_lines(text: str) -> list[str]:
    return [line for line in text.split("\n") if line.startswith("[archex] ")]


# --- observed formats -----------------------------------------------------------

OMP_GREP_SINGLE = """[services/auth.py#8C5F]
 14:
*15:    def login(self, user: User, password: str) -> str:
 16:        token = hash_password(f"{user.id}:{password}")
...
 22:
*23:    def verify_token(self, token: str) -> User | None:
 24:        return self._sessions.get(token)
"""

OMP_GREP_TREE = """# services/
## auth.py#8C5F
 15|    def login(self, user: User, password: str) -> str:
*16|        token = hash_password(f"{user.id}:{password}")
 17|        self._sessions[token] = user

# utils.py#171E
*9|def hash_password(password: str) -> str:
"""

OMP_GREP_BARE = """ 12|    def __init__(self) -> None:
*13|        self._sessions: dict[str, User] = {}
 14|
"""

OMP_GLOB_TREE = """# services/
auth.py
__init__.py
# ./
utils.py
[20 results limit reached. Use limit=40 for more]
"""

PATH_LINE_OUTPUT = """services/auth.py:16:        token = hash_password(f"{user.id}:{password}")
utils.py:9:def hash_password(password: str) -> str:

Wall time: 0.01 seconds
"""

OMP_GLOB_ROOT_ENTRIES = """utils.py
main.py
# services/
auth.py
"""

BASH_GROUPED_OUTPUT = """grep: 3 matches in 2 files

services/auth.py:
  16: token = hash_password(f"{user.id}:{password}")

utils.py:
  9: def hash_password(password: str) -> str:
  10- return hashlib.sha256(password.encode()).hexdigest()
[raw output: artifact://14]
"""


def test_omp_grep_single_file_resolves_each_hit_to_its_method(indexed_repo: Path) -> None:
    result = annotate("grep", {"pattern": "def "}, OMP_GREP_SINGLE, indexed_repo)

    assert result.annotated
    assert result.format == "omp-grep-file"
    assert _unit_lines(result.text) == [
        "[archex] services/auth.py::AuthService.login method L15-18 · importers 1",
    ]
    # verify_token (L23-24) is wholly visible in the result, so it adds nothing.
    assert result.units_hit == 2


def test_omp_grep_tree_resolves_nested_and_depth_one_file_headers(indexed_repo: Path) -> None:
    result = annotate("grep", {"pattern": "hash_password"}, OMP_GREP_TREE, indexed_repo)

    assert result.format == "omp-grep-tree"
    assert _unit_lines(result.text) == [
        "[archex] services/auth.py::AuthService.login method L15-18 · importers 1",
        "[archex] utils.py::hash_password function L9-10 · importers 2",
    ]


def test_omp_grep_bare_output_takes_its_file_from_the_input_path(indexed_repo: Path) -> None:
    result = annotate(
        "grep", {"pattern": "_sessions", "path": "services/auth.py"}, OMP_GREP_BARE, indexed_repo
    )

    assert result.format == "omp-grep-bare"
    # __init__ (L12-13) is fully visible, so only the header would remain.
    assert result.reason == "all_units_visible"
    assert result.text == ""


def test_omp_grep_bare_output_without_a_single_input_path_is_unrecognized(
    indexed_repo: Path,
) -> None:
    result = annotate("grep", {"pattern": "x", "path": "src;tests"}, OMP_GREP_BARE, indexed_repo)

    assert result.reason == "unrecognized_format"
    assert result.text == ""


def test_omp_glob_tree_annotates_each_file_with_units_and_importers(indexed_repo: Path) -> None:
    result = annotate("glob", {"path": "**/*.py"}, OMP_GLOB_TREE, indexed_repo)

    assert result.format == "omp-glob-tree"
    # services/__init__.py holds no code unit, so it gets no line.
    assert _unit_lines(result.text) == [
        "[archex] services/auth.py · units 1 · importers 1",
        "[archex] utils.py · units 2 · importers 2",
    ]


def test_omp_glob_names_before_the_first_header_sit_in_the_search_root(
    indexed_repo: Path,
) -> None:
    result = annotate("glob", {"path": "**/*.py"}, OMP_GLOB_ROOT_ENTRIES, indexed_repo)

    assert _unit_lines(result.text) == [
        "[archex] utils.py · units 2 · importers 2",
        "[archex] main.py · units 1 · importers 0",
        "[archex] services/auth.py · units 1 · importers 1",
    ]


@pytest.mark.parametrize(
    "command",
    [
        'rg -n "hash_password|_sessions" .',
        "grep -rn hash_password .",
        "git grep -n hash_password",
    ],
)
def test_bash_search_path_line_output_is_annotated(indexed_repo: Path, command: str) -> None:
    result = annotate("bash", {"command": command}, PATH_LINE_OUTPUT, indexed_repo)

    assert result.format == "path-line"
    assert _unit_lines(result.text) == [
        "[archex] services/auth.py::AuthService.login method L15-18 · importers 1",
        "[archex] utils.py::hash_password function L9-10 · importers 2",
    ]


def test_bash_grouped_search_view_is_annotated(indexed_repo: Path) -> None:
    result = annotate(
        "bash", {"command": "grep -rn hash_password ."}, BASH_GROUPED_OUTPUT, indexed_repo
    )

    assert result.format == "grouped"
    # hash_password L9-10 is wholly visible (a match and a context entry).
    assert _unit_lines(result.text) == [
        "[archex] services/auth.py::AuthService.login method L15-18 · importers 1",
    ]
    assert result.units_hit == 2


def test_bash_no_output_placeholder_is_a_search_with_no_hits(indexed_repo: Path) -> None:
    result = annotate("bash", {"command": "rg -n nothing ."}, "(no output)\n", indexed_repo)

    assert result.eligible
    assert result.reason == "no_hits"


def test_bash_cd_prefix_resolves_paths_against_the_cd_directory(indexed_repo: Path) -> None:
    output = "auth.py:16:        token = hash_password(...)\n"
    result = annotate(
        "bash", {"command": "cd services && grep -n hash_password auth.py"}, output, indexed_repo
    )

    assert _unit_lines(result.text) == [
        "[archex] services/auth.py::AuthService.login method L15-18 · importers 1",
    ]


def test_bash_context_lines_count_as_visible(indexed_repo: Path) -> None:
    output = (
        "utils.py-8-\n"
        "utils.py:9:def hash_password(password: str) -> str:\n"
        "utils.py-10-    return x\n"
    )
    result = annotate("bash", {"command": "grep -rn -C1 hash_password ."}, output, indexed_repo)

    assert result.reason == "all_units_visible"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('rg -n "a|b" src', ShellSearch("rg", "", "lines", ("src",))),
        ("grep -rn foo . | head -20", ShellSearch("grep", "", "lines", (".",))),
        ("cd sub && git grep -n foo", ShellSearch("git grep", "sub", "lines")),
        ("cd 'with space' && egrep -n foo x", ShellSearch("egrep", "with space", "lines", ("x",))),
        ("/usr/bin/grep -n foo x", ShellSearch("grep", "", "lines", ("x",))),
        ("grep -n -A 3 -e foo -e bar a.py", ShellSearch("grep", "", "lines", ("a.py",))),
        ("grep -nA3 foo a.py b.py", ShellSearch("grep", "", "lines", ("a.py", "b.py"))),
        ("rg -n --glob '*.py' -r X foo -- -dash", ShellSearch("rg", "", "lines", ("-dash",))),
        ("ugrep -n foo a.py", ShellSearch("ugrep", "", "lines", ("a.py",))),
        ("grep -rln foo .", ShellSearch("grep", "", "files")),
        ("rg --files src", ShellSearch("rg", "", "files")),
        ("rg -L -n foo", ShellSearch("rg", "", "lines")),
        ("git grep --name-only foo", ShellSearch("git grep", "", "files")),
        ("find . -name '*.py'", ShellSearch("find", "", "files")),
        ("fd -e py | head", ShellSearch("fd", "", "files")),
        ("git ls-files", ShellSearch("git ls-files", "", "files")),
        ("grep -c foo a.py", None),
        ("rg -q foo", None),
        ("cat x | grep foo", None),
        ("echo grep", None),
        ("git log --grep foo", None),
        ('rg "unterminated', None),
    ],
)
def test_shell_search_classifies_command_and_output(
    command: str, expected: ShellSearch | None
) -> None:
    assert parse_shell_search(command) == expected


def test_non_search_bash_command_is_ineligible(indexed_repo: Path) -> None:
    result = annotate("bash", {"command": "cat utils.py"}, PATH_LINE_OUTPUT, indexed_repo)

    assert not result.eligible
    assert result.reason == "not_search_command"


@pytest.mark.parametrize(
    ("host", "tool", "tool_input", "reason"),
    [
        ("claude-code", "Read", {}, "unsupported_tool"),
        ("claude-code", "grep", {}, "unsupported_tool"),
        ("claude-code", "Bash", {"command": "ls -la"}, "not_search_command"),
        ("codex", "Bash", {"command": ["bash", "-lc", "cargo test"]}, "not_search_command"),
        ("codex", "apply_patch", {}, "unsupported_tool"),
        ("opencode", "Grep", {}, "unsupported_tool"),
        ("cursor", "grep", {}, "unsupported_host"),
    ],
)
def test_calls_that_are_not_searches_on_their_host_are_declined(
    host: str, tool: str, tool_input: dict[str, object], reason: str
) -> None:
    assert classify_call(host, tool, tool_input) == reason


# --- Claude Code, Codex, and OpenCode formats -----------------------------------

LOGIN_LINE = "[archex] services/auth.py::AuthService.login method L15-18 · importers 1"
HASH_LINE = "[archex] utils.py::hash_password function L9-10 · importers 2"

# Claude Code Bash `grep -rn x .` stdout (paths carry the `./` of the operand).
CLAUDE_BASH_DOT = (
    './services/auth.py:16:        token = hash_password(f"{user.id}:{password}")\n'
    "./utils.py:9:def hash_password(password: str) -> str:"
)
# One-file search: `grep -n x utils.py` / `rg -n x utils.py` print no path.
ONE_FILE = "9:def hash_password(password: str) -> str:"


def test_claude_code_bash_grep_output_is_annotated(indexed_repo: Path) -> None:
    result = annotate(
        "Bash",
        {"command": "grep -rn hash_password .", "description": "search"},
        CLAUDE_BASH_DOT,
        indexed_repo,
        host="claude-code",
    )

    assert result.format == "path-line"
    assert _unit_lines(result.text) == [LOGIN_LINE, HASH_LINE]


@pytest.mark.parametrize(
    "command", ["grep -n hash_password utils.py", "rg -n hash_password utils.py"]
)
def test_one_file_search_takes_its_path_from_the_operand(indexed_repo: Path, command: str) -> None:
    result = annotate("Bash", {"command": command}, ONE_FILE, indexed_repo, host="claude-code")

    assert result.format == "line-numbered"
    assert _unit_lines(result.text) == [HASH_LINE]


def test_one_file_context_entries_count_as_visible(indexed_repo: Path) -> None:
    output = "9:def hash_password(password: str) -> str:\n10-    return x"
    result = annotate(
        "Bash",
        {"command": "grep -n -A1 hash_password utils.py"},
        output,
        indexed_repo,
        host="codex",
    )

    assert result.reason == "all_units_visible"


def test_line_entries_without_a_single_operand_are_unrecognized(indexed_repo: Path) -> None:
    result = annotate(
        "Bash",
        {"command": "grep -hn hash_password utils.py main.py"},
        ONE_FILE,
        indexed_repo,
        host="claude-code",
    )

    assert result.reason == "unrecognized_format"


@pytest.mark.parametrize(
    ("command", "output"),
    [
        ("find . -name '*.py'", "./utils.py\n./services/auth.py\n./services/__init__.py"),
        ("rg --files", "utils.py\nservices/auth.py"),
        ("grep -rl hash_password .", "./utils.py\n./services/auth.py"),
    ],
)
def test_shell_path_listings_annotate_each_file(
    indexed_repo: Path, command: str, output: str
) -> None:
    result = annotate("Bash", {"command": command}, output, indexed_repo, host="claude-code")

    assert result.format == "path-list"
    assert _unit_lines(result.text) == [
        "[archex] utils.py · units 2 · importers 2",
        "[archex] services/auth.py · units 1 · importers 1",
    ]


def test_claude_code_grep_content_mode_is_annotated(indexed_repo: Path) -> None:
    tool_input = {"pattern": "hash_password", "output_mode": "content", "-n": True}
    content = CLAUDE_BASH_DOT.replace("./", "")

    result = annotate("Grep", tool_input, content, indexed_repo, host="claude-code")

    assert _unit_lines(result.text) == [LOGIN_LINE, HASH_LINE]


def test_claude_code_grep_on_one_file_takes_its_path_from_the_input(indexed_repo: Path) -> None:
    tool_input = {"pattern": "hash_password", "path": "utils.py", "output_mode": "content"}

    result = annotate("Grep", tool_input, ONE_FILE, indexed_repo, host="claude-code")

    assert result.format == "line-numbered"
    assert _unit_lines(result.text) == [HASH_LINE]


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [("Grep", {"pattern": "hash_password"}), ("Glob", {"pattern": "**/*.py"})],
)
def test_claude_code_file_lists_annotate_each_file(
    indexed_repo: Path, tool: str, tool_input: dict[str, object]
) -> None:
    filenames = "utils.py\nservices/auth.py"

    result = annotate(tool, tool_input, filenames, indexed_repo, host="claude-code")

    assert result.format == "path-list"
    assert len(_unit_lines(result.text)) == 2


def test_claude_code_grep_count_mode_adds_nothing(indexed_repo: Path) -> None:
    tool_input = {"pattern": "hash_password", "output_mode": "count"}

    result = annotate("Grep", tool_input, "utils.py:1", indexed_repo, host="claude-code")

    assert result.reason == "count_output"
    assert result.text == ""


def test_codex_argv_shell_command_is_read_through_its_shell(indexed_repo: Path) -> None:
    tool_input = {"command": ["bash", "-lc", "rg -n hash_password utils.py"]}

    result = annotate("Bash", tool_input, ONE_FILE, indexed_repo, host="codex")

    assert _unit_lines(result.text) == [HASH_LINE]


def test_opencode_grep_output_is_annotated(indexed_repo: Path) -> None:
    root = indexed_repo.resolve()
    output = (
        "Found 2 matches\n"
        f"{root}/services/auth.py:\n"
        '  Line 16:         token = hash_password(f"{user.id}:{password}")\n'
        "\n"
        f"{root}/utils.py:\n"
        "  Line 9: def hash_password(password: str) -> str:"
    )

    result = annotate("grep", {"pattern": "hash_password"}, output, indexed_repo, host="opencode")

    assert result.format == "opencode-grep"
    assert _unit_lines(result.text) == [LOGIN_LINE, HASH_LINE]


def test_opencode_glob_output_skips_its_truncation_note(indexed_repo: Path) -> None:
    root = indexed_repo.resolve()
    output = (
        f"{root}/utils.py\n{root}/main.py\n\n"
        "(Results are truncated: showing first 2 results. Consider using a more specific "
        "path or pattern.)"
    )

    result = annotate("glob", {"pattern": "**/*.py"}, output, indexed_repo, host="opencode")

    assert _unit_lines(result.text) == [
        "[archex] utils.py · units 2 · importers 2",
        "[archex] main.py · units 1 · importers 0",
    ]


def test_opencode_bash_workdir_and_metadata_block(indexed_repo: Path) -> None:
    output = (
        '16:        token = hash_password(f"{user.id}:{password}")\n\n'
        "<bash_metadata>\nUser aborted the command\n</bash_metadata>"
    )
    tool_input = {"command": "grep -n hash_password auth.py", "workdir": "services"}

    result = annotate("bash", tool_input, output, indexed_repo, host="opencode")

    assert result.format == "line-numbered"
    assert _unit_lines(result.text) == [LOGIN_LINE]


def test_non_search_call_is_declined_without_loading_the_index_or_tokenizer() -> None:
    probe = (
        "import json, sys\n"
        "from archex.integrations.annotate_hook import parse_request, run_request\n"
        "request = parse_request(json.dumps("
        "{'host': 'claude-code', 'tool': 'Bash', 'input': {'command': 'ls -la'}, "
        "'text': 'x', 'cwd': '.'}))\n"
        "assert run_request(request).reason == 'not_search_command'\n"
        "heavy = ('archex.index.store', 'archex.reporting', 'archex.status', 'pydantic', 'click')\n"
        "print(json.dumps(sorted(m for m in heavy if m in sys.modules)))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert json.loads(completed.stdout) == []


# --- resolution boundaries ------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (15, "services/auth.py::AuthService.login method L15-18"),
        (18, "services/auth.py::AuthService.login method L15-18"),
        (19, "services/auth.py::AuthService class L11-24"),
        (11, "services/auth.py::AuthService class L11-24"),
        (5, "services/auth.py module-level · units 1"),
    ],
)
def test_hit_resolves_to_the_smallest_unit_containing_it(
    indexed_repo: Path, line: int, expected: str
) -> None:
    output = f"services/auth.py:{line}:x\n"
    result = annotate("bash", {"command": "grep -rn x ."}, output, indexed_repo)

    assert _unit_lines(result.text)[0].startswith(f"[archex] {expected}")


def test_line_past_every_chunk_but_inside_the_file_is_module_level(indexed_repo: Path) -> None:
    result = annotate("bash", {"command": "grep -rn run ."}, "main.py:19:    run()\n", indexed_repo)

    assert _unit_lines(result.text) == ["[archex] main.py module-level · units 1 · importers 0"]


def test_line_past_the_end_of_the_file_is_counted_out_of_range(indexed_repo: Path) -> None:
    result = annotate("bash", {"command": "grep -rn x ."}, "main.py:999:x\n", indexed_repo)

    assert result.reason == "hits_out_of_range"
    assert result.out_of_range_hits == 1


def test_hits_in_one_unit_collapse_to_one_line(indexed_repo: Path) -> None:
    output = "services/auth.py:16:a\nservices/auth.py:17:b\nservices/auth.py:18:c\n"
    result = annotate("bash", {"command": "grep -rn x ."}, output, indexed_repo)

    assert result.units_hit == 1
    assert len(_unit_lines(result.text)) == 1


def test_paths_outside_the_repository_are_ignored(indexed_repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    result = annotate("bash", {"command": "grep -rn x /"}, f"{outside}:1:x = 1\n", indexed_repo)

    assert result.reason == "no_indexed_paths"


# --- caps -----------------------------------------------------------------------


@pytest.fixture
def many_units_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "many"
    repo.mkdir()
    body = "".join(
        f"def function_number_{i}(value):\n    return value + {i}\n\n\n" for i in range(60)
    )
    (repo / "module.py").write_text(body, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")
    _index(repo)
    return repo


def test_call_is_capped_and_excess_units_collapse(many_units_repo: Path) -> None:
    output = "".join(f"module.py:{1 + 4 * i}:x\n" for i in range(60))
    result = annotate("bash", {"command": "grep -rn x ."}, output, many_units_repo)

    lines = result.text.split("\n")
    assert result.capped
    assert result.units_hit == 60
    assert result.tokens == count_tokens(result.text) <= MAX_CALL_TOKENS
    assert lines[-1] == f"[archex] +{60 - result.units_rendered} more units"
    assert all(count_tokens(line) <= MAX_LINE_TOKENS for line in lines)


# --- freshness, unknown format, determinism -------------------------------------


def test_stale_index_produces_no_annotation(indexed_repo: Path) -> None:
    (indexed_repo / "new.py").write_text("x = 1\n", encoding="utf-8")
    _git(indexed_repo, "add", "new.py")
    _git(indexed_repo, "commit", "-qm", "advance")

    result = annotate("bash", {"command": "grep -rn x ."}, PATH_LINE_OUTPUT, indexed_repo)

    assert result.text == ""
    assert result.freshness == "stale"
    assert result.reason == "index_not_fresh"


def test_dirty_index_produces_no_annotation(indexed_repo: Path) -> None:
    (indexed_repo / "utils.py").write_text("x = 1\n", encoding="utf-8")

    result = annotate("bash", {"command": "grep -rn x ."}, PATH_LINE_OUTPUT, indexed_repo)

    assert result.text == ""
    assert result.freshness == "dirty"


def test_unknown_grep_format_is_declined_before_the_index_is_read() -> None:
    parsed = parse_search_result("grep", {"pattern": "x"}, "something else entirely\n")

    assert parsed == "unrecognized_format"


def test_same_index_and_hits_give_byte_identical_annotations(indexed_repo: Path) -> None:
    first = annotate("grep", {"pattern": "x"}, OMP_GREP_TREE, indexed_repo)
    second = annotate("grep", {"pattern": "x"}, OMP_GREP_TREE, indexed_repo)

    assert first.text
    assert first.text == second.text
    assert first.text.split("\n")[0].startswith("[archex receipt] index_revision=")


# --- CLI ------------------------------------------------------------------------


def test_cli_prints_only_the_annotation(indexed_repo: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["annotate", "--tool", "grep", "--input-json", "{}", "--cwd", str(indexed_repo)],
        input=OMP_GREP_TREE,
    )

    assert result.exit_code == 0
    expected = annotate("grep", {}, OMP_GREP_TREE, indexed_repo).text
    assert result.output == expected + "\n"


def test_cli_json_envelope_reports_the_decision(indexed_repo: Path) -> None:
    envelope = {"tool": "bash", "input": {"command": "ls"}, "text": "x", "cwd": str(indexed_repo)}
    result = CliRunner().invoke(
        cli, ["annotate", "--stdin-json", "--format", "json"], input=json.dumps(envelope)
    )

    record = json.loads(result.output)
    assert result.exit_code == 0
    assert record["eligible"] is False
    assert record["reason"] == "not_search_command"


def test_cli_logs_unrecognized_format_and_prints_nothing(
    indexed_repo: Path, diagnostics_log: Path
) -> None:
    result = CliRunner().invoke(
        cli,
        ["annotate", "--tool", "grep", "--cwd", str(indexed_repo)],
        input="not a grep result\n",
    )

    assert result.exit_code == 0
    assert result.output == ""
    entry = json.loads(diagnostics_log.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["kind"] == "annotate_declined"
    assert "reason=unrecognized_format" in entry["detail"]


@pytest.mark.parametrize(
    "stdin", ["not json", "[]", '{"tool": 3}', '{"tool": "grep", "input": []}']
)
def test_cli_malformed_envelope_fails_open(
    tmp_path: Path, diagnostics_log: Path, stdin: str
) -> None:
    result = CliRunner().invoke(
        cli, ["annotate", "--stdin-json", "--format", "json", "--cwd", str(tmp_path)], input=stdin
    )

    assert result.exit_code == 0
    assert result.output == ""
    entry = json.loads(diagnostics_log.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["kind"] == "annotate_error"
