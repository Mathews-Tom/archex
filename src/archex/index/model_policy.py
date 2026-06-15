"""Central model-loading security policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from archex.exceptions import ConfigError

ModelComponent = Literal["embedder", "reranker"]

JINA_V2_MODEL_ID = "jinaai/jina-embeddings-v2-base-code"
JINA_V2_MODEL_REVISION = "516f4baf13dec4ddddda8631e019b5737c8bc250"
JINA_BERT_CODE_REVISION = "3baf9e3ac750e76e8edd3019170176884695fb94"
JINA_V2_MAX_SEQ_LENGTH = 1024
NOMIC_CODE_MODEL_ID = "nomic-ai/nomic-embed-code"
NOMIC_CODE_MODEL_REVISION = "11114029805cee545ef111d5144b623787462a52"
CODERANK_MODEL_ID = "nomic-ai/CodeRankEmbed"
CODERANK_MODEL_REVISION = "3c4b60807d71f79b43f3c4363786d9493691f8b1"
JINA_RERANKER_MODEL = "jinaai/jina-reranker-v3"
JINA_RERANKER_REVISION = "10fb694fc21f7a710a563ff1eb977a460f3868e4"

REMOTE_CODE_OPT_IN_HINT = (
    "Set allow_remote_code = true under [index] in .archex/settings.toml, "
    "export ARCHEX_ALLOW_REMOTE_CODE=true, or pass --allow-remote-code on supported CLI commands."
)


@dataclass(frozen=True)
class ModelSecurityProfile:
    component: ModelComponent
    provider: str
    model_name: str
    requires_remote_code: bool
    model_revision: str | None = None
    code_revision: str | None = None

    @property
    def revision_label(self) -> str:
        if self.model_revision is None and self.code_revision is None:
            return "unpinned"
        if self.code_revision is None:
            return f"model={self.model_revision}"
        return f"model={self.model_revision}; code={self.code_revision}"


_EMBEDDER_PROFILES: dict[str, ModelSecurityProfile] = {
    "fastembed": ModelSecurityProfile(
        component="embedder",
        provider="fastembed",
        model_name="BAAI/bge-small-en-v1.5",
        requires_remote_code=False,
    ),
    "sentence_transformers": ModelSecurityProfile(
        component="embedder",
        provider="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        requires_remote_code=False,
    ),
    "jina-v2": ModelSecurityProfile(
        component="embedder",
        provider="sentence-transformers",
        model_name=JINA_V2_MODEL_ID,
        requires_remote_code=True,
        model_revision=JINA_V2_MODEL_REVISION,
        code_revision=JINA_BERT_CODE_REVISION,
    ),
    "nomic": ModelSecurityProfile(
        component="embedder",
        provider="sentence-transformers",
        model_name=NOMIC_CODE_MODEL_ID,
        requires_remote_code=True,
        model_revision=NOMIC_CODE_MODEL_REVISION,
    ),
    "coderank": ModelSecurityProfile(
        component="embedder",
        provider="sentence-transformers",
        model_name=CODERANK_MODEL_ID,
        requires_remote_code=True,
        model_revision=CODERANK_MODEL_REVISION,
    ),
}

_RERANKER_PROFILES: dict[str, ModelSecurityProfile] = {
    JINA_RERANKER_MODEL: ModelSecurityProfile(
        component="reranker",
        provider="transformers",
        model_name=JINA_RERANKER_MODEL,
        requires_remote_code=True,
        model_revision=JINA_RERANKER_REVISION,
    ),
}


def embedder_security_profile(embedder: str | None) -> ModelSecurityProfile:
    name = embedder or "sentence_transformers"
    return _EMBEDDER_PROFILES.get(
        name,
        ModelSecurityProfile(
            component="embedder",
            provider="entry-point",
            model_name=name,
            requires_remote_code=False,
        ),
    )


def reranker_security_profile(model_name: str) -> ModelSecurityProfile:
    return _RERANKER_PROFILES.get(
        model_name,
        ModelSecurityProfile(
            component="reranker",
            provider="sentence-transformers",
            model_name=model_name,
            requires_remote_code=False,
        ),
    )


def custom_remote_code_profile(
    *,
    component: ModelComponent,
    provider: str,
    model_name: str,
    model_revision: str | None,
    code_revision: str | None = None,
) -> ModelSecurityProfile:
    return ModelSecurityProfile(
        component=component,
        provider=provider,
        model_name=model_name,
        requires_remote_code=True,
        model_revision=model_revision,
        code_revision=code_revision,
    )


def require_remote_code_opt_in(
    profile: ModelSecurityProfile,
    *,
    allow_remote_code: bool,
) -> None:
    """Reject remote-code model loading unless the caller explicitly opted in."""
    if not profile.requires_remote_code or allow_remote_code:
        return
    raise ConfigError(
        f"Remote code is disabled for {profile.component} provider "
        f"'{profile.provider}' model '{profile.model_name}'. "
        "This model path requires trust_remote_code=True. "
        f"Pinned revisions: {profile.revision_label}. {REMOTE_CODE_OPT_IN_HINT}"
    )


def remote_code_trust_value(profile: ModelSecurityProfile, *, allow_remote_code: bool) -> bool:
    """Return the trust_remote_code value after enforcing the opt-in policy."""
    require_remote_code_opt_in(profile, allow_remote_code=allow_remote_code)
    return profile.requires_remote_code and allow_remote_code
