"""Explicit indexing command for repo-local archex project workflows."""

from __future__ import annotations

import json
from pathlib import Path

import click

from archex.cli.indexing import run_indexing_and_get_summary
from archex.exceptions import ArchexError


@click.command("index")
@click.argument("source", required=False, default=".")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format.",
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
def index_cmd(
    source: str,
    output_format: str,
    splade: bool,
    module_prefilter: bool,
    allow_remote_code: bool,
    quantize_vectors: bool | None,
    quantize_bits: str | None,
    export_artifact_path: Path | None,
) -> None:
    """Build or refresh the index for SOURCE without running a query."""
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

    if output_format == "json":
        click.echo(json.dumps(summary, indent=2, sort_keys=True))
        return

    click.echo(f"Indexed repository: {summary['repo_root']}")
    click.echo(f"Index path:         {summary['index_path']}")
    click.echo(f"Commit:             {summary['commit_hash'] or 'none'}")
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
    click.echo(
        "Embedding cache:    "
        f"hits={summary['embedding_cache_hits']}, misses={summary['embedding_cache_misses']}"
    )
    click.echo(f"Duration:           {summary['duration_ms']} ms")
    if export_artifact_path is not None:
        click.echo(f"Artifact exported:  {summary['artifact_path']}")
        click.echo(f"Artifact size:      {summary['artifact_size_bytes']} bytes")
        if summary["gitattributes_updated"]:
            click.echo(f"Updated .gitattributes: {Path(summary['repo_root']) / '.gitattributes'}")
