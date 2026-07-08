import json
import subprocess
from pathlib import Path

from click.testing import CliRunner

from archex.cli.main import cli


def test_setup_dry_run(tmp_path: Path) -> None:
    subprocess.check_call(["git", "init", str(tmp_path)])
    runner = CliRunner()
    result = runner.invoke(cli, ["setup", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0
    assert "--- Setup Preflight ---" in result.output
    assert "has_dot_archex: False" in result.output


def test_setup_json_format(tmp_path: Path) -> None:
    subprocess.check_call(["git", "init", str(tmp_path)])
    runner = CliRunner()
    result = runner.invoke(cli, ["setup", str(tmp_path), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "preflight" in data
    assert "planned_actions" in data
    assert data["preflight"]["has_dot_archex"] is False
    assert data["preflight"]["has_index"] is False


def test_setup_non_tty_error(tmp_path: Path) -> None:
    subprocess.check_call(["git", "init", str(tmp_path)])
    runner = CliRunner()
    # CliRunner simulates non-TTY
    result = runner.invoke(cli, ["setup", str(tmp_path)])
    assert result.exit_code == 1
    assert "setup is interactive by default" in result.output


def test_setup_init_index(tmp_path: Path) -> None:
    subprocess.check_call(["git", "init", str(tmp_path)])
    runner = CliRunner()
    result = runner.invoke(cli, ["setup", str(tmp_path), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "planned_actions" in data
    actions = data["planned_actions"]
    init_action = next(a for a in actions if a["type"] == "init")
    assert init_action["status"] == "planned"

    index_action = next(a for a in actions if a["type"] == "index")
    assert index_action["status"] == "planned"


def test_setup_yes_executes(tmp_path: Path) -> None:
    subprocess.check_call(["git", "init", str(tmp_path)])
    # Create at least one file so indexing works
    (tmp_path / "hello.py").write_text("print('hello')")
    runner = CliRunner()
    result = runner.invoke(cli, ["setup", str(tmp_path), "--yes"])
    assert result.exit_code == 0
    assert "Executing Setup" in result.output
    assert "init: executed" in result.output
    assert "index: executed" in result.output
