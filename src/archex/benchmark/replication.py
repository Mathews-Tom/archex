"""Schema and validation for external-replication evidence artifacts.

A replication artifact records an attempt to reproduce a *published* result in
the original paper's own reference setup. It is deliberately not the same shape
as an archex benchmark evidence directory: there are no archex task IDs, no
archex strategies, and no archex source revision governing the measurement, so
:func:`archex.benchmark.evidence.validate_evidence_directory` cannot speak to it.

The schema exists to make three failure modes loud rather than silent:

* an unpinned reproduction, which cannot be rerun and therefore proves nothing;
* an unlabelled arm, which lets a replication-class claim drift into an
  adaptation-class one; and
* a verdict that does not follow from the numbers it is filed against.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ReplicationEvidenceError(ValueError):
    """Raised when a replication artifact cannot be trusted to mean what it says."""


class EvidenceClass(StrEnum):
    REPLICATION = "replication"
    ADAPTATION = "adaptation"
    ORIGINAL = "original"


class ReplicationVerdict(StrEnum):
    """The four dispositions an arm may carry. Only ``PASS`` clears a gate."""

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    UNRUNNABLE = "unrunnable"


class ReplicationSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReplicationPins(ReplicationSpecModel):
    """Every immutable input a rerun needs. A missing pin is a validation error."""

    harness_repo: str = Field(min_length=1)
    harness_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dataset: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    dataset_split: str = Field(min_length=1)
    models: dict[str, str] = Field(min_length=1)
    environment: dict[str, str] = Field(min_length=1)
    command: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_models(self) -> ReplicationPins:
        for label, mapping in (("models", self.models), ("environment", self.environment)):
            for key, value in mapping.items():
                if not key.strip():
                    msg = f"{label} must not contain an empty key"
                    raise ValueError(msg)
                if not value.strip():
                    msg = f"{label} entry {key!r} must pin a version"
                    raise ValueError(msg)
        return self


class ReplicationInterval(ReplicationSpecModel):
    """A two-sided interval, low first."""

    low: float
    high: float
    method: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_order(self) -> ReplicationInterval:
        if self.low > self.high:
            msg = f"interval low {self.low} exceeds high {self.high}"
            raise ValueError(msg)
        return self

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


def derive_verdict(
    *,
    delta: float,
    band_low: float,
    band_high: float,
    ci_low: float,
    ci_high: float,
) -> ReplicationVerdict:
    """The pre-registered decision rule, in one place.

    The equivalence band is **closed**: a delta exactly on an edge is inside it.
    The significance test is **open**: an interval touching zero is not
    significant, which is the conservative direction. Both the schema and the
    analysis call this, so the two can never drift apart.
    """
    if not band_low <= delta <= band_high:
        return ReplicationVerdict.FAIL
    significant = ci_low > 0.0 or ci_high < 0.0
    return ReplicationVerdict.PASS if significant else ReplicationVerdict.INCONCLUSIVE


class ReplicationArm(ReplicationSpecModel):
    """One reproduction attempt against one published cell."""

    arm_id: str = Field(min_length=1)
    evidence_class: EvidenceClass
    paper: str = Field(min_length=1)
    paper_cell: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    reported_delta: float | None = None
    equivalence_band: ReplicationInterval | None = None
    reproduced_delta: float | None = None
    reproduced_interval: ReplicationInterval | None = None
    verdict: ReplicationVerdict
    rationale: str = Field(min_length=1)
    pins: ReplicationPins | None = None

    @model_validator(mode="after")
    def _validate_verdict_support(self) -> ReplicationArm:
        if self.verdict is ReplicationVerdict.UNRUNNABLE:
            if self.reproduced_delta is not None:
                msg = f"arm {self.arm_id!r} is unrunnable but reports a reproduced delta"
                raise ValueError(msg)
            return self

        missing = [
            name
            for name, value in (
                ("reported_delta", self.reported_delta),
                ("equivalence_band", self.equivalence_band),
                ("reproduced_delta", self.reproduced_delta),
                ("reproduced_interval", self.reproduced_interval),
                ("pins", self.pins),
            )
            if value is None
        ]
        if missing:
            msg = f"arm {self.arm_id!r} has verdict {self.verdict.value!r} but omits {missing}"
            raise ValueError(msg)

        band = self.equivalence_band
        interval = self.reproduced_interval
        delta = self.reproduced_delta
        reported = self.reported_delta
        if band is None or interval is None or delta is None or reported is None:
            msg = f"arm {self.arm_id!r} lost a required field after the completeness check"
            raise ValueError(msg)

        if not band.contains(reported):
            msg = (
                f"arm {self.arm_id!r} has an equivalence band [{band.low}, {band.high}] "
                f"that does not contain the reported delta {reported} it is supposed to "
                "be centred on"
            )
            raise ValueError(msg)
        if not interval.contains(delta):
            msg = (
                f"arm {self.arm_id!r} reports a delta of {delta} that lies outside its "
                f"own interval [{interval.low}, {interval.high}]"
            )
            raise ValueError(msg)
        if reported != 0.0 and not band.excludes_zero():
            msg = (
                f"arm {self.arm_id!r} has a nonzero reported delta {reported} "
                f"but a band [{band.low}, {band.high}] straddling zero, which would let a "
                "sign-flipped reproduction be recorded as a pass"
            )
            raise ValueError(msg)

        implied = derive_verdict(
            delta=delta,
            band_low=band.low,
            band_high=band.high,
            ci_low=interval.low,
            ci_high=interval.high,
        )
        if self.verdict is not implied:
            msg = (
                f"arm {self.arm_id!r} records verdict {self.verdict.value!r}, but a reproduced "
                f"delta of {delta} against band [{band.low}, {band.high}] with interval "
                f"[{interval.low}, {interval.high}] implies {implied.value!r}"
            )
            raise ValueError(msg)
        return self


class ReplicationArtifact(ReplicationSpecModel):
    """The checked-in record of one replication spike."""

    replication_version: Literal[1] = 1
    spike_id: str = Field(min_length=1)
    preregistration: str = Field(min_length=1)
    preregistration_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    generated_at: str = Field(min_length=1)
    hardware: str = Field(min_length=1)
    arms: list[ReplicationArm] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_arms(self) -> ReplicationArtifact:
        arm_ids = [arm.arm_id for arm in self.arms]
        if len(arm_ids) != len(set(arm_ids)):
            msg = "arms must not contain duplicate arm_id values"
            raise ValueError(msg)
        return self

    @property
    def passing_arms(self) -> list[ReplicationArm]:
        return [arm for arm in self.arms if arm.verdict is ReplicationVerdict.PASS]


def validate_replication_artifact(path: Path) -> ReplicationArtifact:
    """Load and validate one replication artifact, failing loudly on any defect."""
    if not path.is_file():
        msg = f"Replication artifact is not a file: {path}"
        raise ReplicationEvidenceError(msg)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Replication artifact is not readable JSON: {path}: {exc}"
        raise ReplicationEvidenceError(msg) from exc
    try:
        return ReplicationArtifact.model_validate(payload)
    except ValidationError as exc:
        msg = f"Replication artifact failed validation: {path}: {exc}"
        raise ReplicationEvidenceError(msg) from exc
