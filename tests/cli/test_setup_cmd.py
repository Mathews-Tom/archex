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
