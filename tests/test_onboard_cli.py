from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from archex.cli.main import cli


def test_onboard_outputs_deterministic_markdown(python_simple_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["onboard", str(python_simple_repo), "--max-files", "5"])

    assert result.exit_code == 0, result.output
    assert result.output.startswith("# Onboarding:")
    assert "## Repository Overview" in result.output
    assert "## Recommended Reading Order" in result.output
    assert "`main.py`" in result.output
    assert "`pyproject.toml`" in result.output


def test_onboard_writes_output_file(python_simple_repo: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    output_path = tmp_path / "ONBOARDING.md"

    result = runner.invoke(
        cli,
        ["onboard", str(python_simple_repo), "--output", str(output_path), "--max-files", "3"],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == str(output_path)
    rendered = output_path.read_text(encoding="utf-8")
    assert "## Architecture Modules" in rendered
    assert "## Generated Artifact Metadata" in rendered


def test_onboard_rejects_invalid_max_files(python_simple_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["onboard", str(python_simple_repo), "--max-files", "0"])

    assert result.exit_code != 0
    assert "max-files must be greater than zero" in result.output
