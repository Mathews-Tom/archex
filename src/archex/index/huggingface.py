"""Hugging Face model resolution helpers."""

from __future__ import annotations

from pathlib import Path

from archex.exceptions import ArchexIndexError


def resolve_hf_model_path(model_name: str, *, revision: str | None = None) -> str:
    """Resolve a Hugging Face model id to a local snapshot path.

    Prefer an already-cached snapshot so downstream tokenizers/models receive a
    local path and avoid incidental Hugging Face metadata calls during offline
    benchmark runs. If the snapshot is absent, perform the normal download path.
    """
    local_path = Path(model_name).expanduser()
    if local_path.exists():
        return str(local_path)

    try:
        from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]
    except ImportError as exc:
        raise ArchexIndexError(
            "Hugging Face model loading requires huggingface-hub. "
            "Install the vector-torch or splade extras."
        ) from exc

    try:
        return str(
            snapshot_download(
                repo_id=model_name,
                revision=revision,
                local_files_only=True,
            )
        )
    except Exception:
        try:
            return str(
                snapshot_download(
                    repo_id=model_name,
                    revision=revision,
                    local_files_only=False,
                )
            )
        except Exception as download_exc:
            message = (
                f"Unable to load Hugging Face model '{model_name}'. "
                "Cache it locally first or run with network access enabled."
            )
            if revision is not None:
                message += f" Requested revision: {revision}."
            raise ArchexIndexError(message) from download_exc
