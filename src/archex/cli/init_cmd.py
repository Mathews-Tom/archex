"""Project lifecycle initialization command."""

from __future__ import annotations

from pathlib import Path

import click

from archex.cli.indexing import run_indexing_and_get_summary
from archex.config import load_config, load_index_config
from archex.exceptions import ArchexError
from archex.index.artifact import import_artifact, sync_imported_artifact
from archex.project import init_project


@click.command("init")
@click.argument("source", required=False, default=".")
@click.option("--force", is_flag=True, default=False, help="Rewrite project settings.")
@click.option(
    "--reset",
    is_flag=True,
    default=False,
    help="Delete existing .archex state before initialization. Requires --force.",
)
@click.option(
    "--from-artifact",
    "from_artifact_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Bootstrap the local index from a portable index artifact instead of cold-start reindex.",
)
@click.option(
    "--index/--no-index",
    is_flag=True,
    default=True,
    help="Build or refresh the repository index after initialization (default: true).",
)
@click.option("--splade", is_flag=True, default=False, help="Build the opt-in SPLADE index.")
@click.option(
    "--module-prefilter",
    is_flag=True,
    default=False,
    help="Build opt-in module responsibility summaries.",
)
@click.option(
    "--allow-remote-code",
    is_flag=True,
    default=False,
    help="Allow explicitly selected pinned model paths that require Hugging Face remote code.",
)
@click.option(
    "--quantize-vectors/--no-quantize-vectors",
    default=None,
    help="Build vector indexes with TurboQuant compression.",
)
@click.option(
    "--quantize-bits",
    default=None,
    type=click.Choice(["2", "4"]),
    help="TurboQuant bit-width for vector indexes.",
)
@click.option(
    "--export-artifact",
    "export_artifact_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Export a compacted, compressed, portable index artifact to PATH after indexing.",
)
def init_cmd(
    source: str,
    force: bool,
    reset: bool,
    from_artifact_path: Path | None,
    index: bool,
    splade: bool,
    module_prefilter: bool,
    allow_remote_code: bool,
    quantize_vectors: bool | None,
    quantize_bits: str | None,
    export_artifact_path: Path | None,
) -> None:
    """Initialize repo-local archex project state."""
    try:
        result = init_project(source, force=force, reset=reset)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if result.created:
        click.echo(f"Initialized archex project at {result.state.project_dir}")
    else:
        click.echo(f"archex project already initialized at {result.state.project_dir}")

    if result.settings_written:
        click.echo(f"Wrote settings: {result.state.settings_path}")
    else:
        click.echo(f"Preserved settings: {result.state.settings_path}")

    if result.gitignore_updated:
        click.echo(f"Updated ignore rules: {result.state.gitignore_path}")

    if from_artifact_path is not None:
        try:
            header = import_artifact(from_artifact_path, result.state.index_path)
        except ArchexError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Imported index artifact: {from_artifact_path}")
        click.echo(f"Artifact revision:       {header.index_revision}")
        click.echo(
            f"Artifact corpus:         {header.file_count} files, {header.chunk_count} chunks"
        )

        try:
            config = load_config(source)
            index_config = load_index_config(source)
            sync_result = sync_imported_artifact(
                result.state.repo_root, result.state.index_path, config, index_config
            )
        except ArchexError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Delta-sync strategy:     {sync_result.strategy}")
        if sync_result.strategy != "clean":
            click.echo(f"Files changed since export: {sync_result.files_changed}")
        click.echo(f"Sync time:               {sync_result.sync_time_ms} ms")
    elif index:
        try:
            summary = run_indexing_and_get_summary(
                source=source,
                splade=splade,
                module_prefilter=module_prefilter,
                allow_remote_code=allow_remote_code,
                quantize_vectors=quantize_vectors,
                quantize_bits=quantize_bits,
                export_artifact_path=export_artifact_path,
            )
        except ArchexError as exc:
            raise click.ClickException(str(exc)) from exc

        click.echo(f"Indexed repository: {summary['repo_root']}")
        click.echo(f"Strategy:           {summary['strategy']}")
        click.echo(f"Files indexed:      {summary['files_indexed']}")
        click.echo(f"Chunks indexed:     {summary['chunks_indexed']}")
        if summary["languages"]:
            language_summary = ", ".join(
                f"{language}={count}" for language, count in summary["languages"].items()
            )
        else:
            language_summary = "none"
        click.echo(f"Languages:          {language_summary}")
