"""Treatment manifests for the ratified comparisons (experiment.md).

Vocabulary used in contracts.TreatmentManifest fields (proposed to the
execution session; the runner reconstructs from these):

component_donors: {"workspace": <checkpoint_id>, "runtime": <checkpoint_id>,
                   "history": <checkpoint_id> | "fresh"}
history_policy:   "native"    keep H exactly as at S1 (the runner restores from source)
                  "fresh"     canonical restart scaffold (system + fixed S1 target +
                              orientation) and nothing else; matches mining/runner.py
                  "o_masked"  native H with declared tool-result contents replaced
                              by a fixed placeholder (runner support requested)
runtime_policy:   "restore_captured" restore R from the runtime donor (runner name);
                  "discard_captured" runtime ablation: R_CAPTURED at S1 but deliberately not restored;
                  "irrelevant"       R_IRRELEVANT declared by the task; no runtime key;
                  "unsupported"      branch structurally invalid for R claims
packet_refs:      exact injected bytes (at most one packet per branch)
metadata.packet:  {"kind": none|summary|t_typed|t_prose|t_view:<filter>,
                   "packet_id", "source_version_id", "approx_token_count"}
                  manual only: {"kind": "research_packet", "label", "packet_id", ...}
                  (registered external packet; see research_packets.py)
metadata.mask:    O-masking spec when history_policy == "o_masked"
metadata.study_cells: which study cells reuse this rollout (no double counting)

Every manifest binds one source S1, a fixed target manifest frozen at S1, the
originating worker model, an allowance and a repeat id; branches never
inherit another branch's results because donors are always mining
checkpoints of the source run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from ..contracts import (ArtifactRef, ArtifactStoreProtocol, Checkpoint, TreatmentManifest,
                         canonical_json, worker_request_policy_known)
from ..observers._canonical import content_id
from ..observers.pipeline import FrozenPackets
from .observation_mask import MASK_PLACEHOLDER, validate_observation_mask

HISTORY_POLICIES = ("native", "fresh", "o_masked")
RUNTIME_POLICIES = ("restore_captured", "irrelevant", "unsupported", "discard_captured")
ORIENTATION_TEXT = "The workspace may contain prior work. Inspect it as needed."


def recovery_settings_known(value: Any) -> bool:
    """No operational defaults are inferred for an undeclared recovery."""
    return (isinstance(value, Mapping)
            and "max_model_calls" in value
            and (value["max_model_calls"] is None or
                 (type(value["max_model_calls"]) is int and value["max_model_calls"] > 0))
            and type(value.get("tool_timeout_seconds")) in (int, float)
            and math.isfinite(value["tool_timeout_seconds"]) and value["tool_timeout_seconds"] > 0
            and worker_request_policy_known(value.get("worker_request_policy"))
            and type(value.get("max_consecutive_format_errors")) is int and value["max_consecutive_format_errors"] >= 0
            and isinstance(value.get("detector_settings"), Mapping))


@dataclass(frozen=True)
class RecoveryTarget:
    """Fixed recovery target active at S1 (Q16)."""

    target_id: str
    stage_id: str
    requirement_version: str
    evaluator_version: str
    prerequisites: tuple[str, ...]
    instructions_ref: ArtifactRef
    completion_condition: str = "all_active_checks_pass"

    def as_record(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id, "stage_id": self.stage_id,
            "requirement_version": self.requirement_version, "evaluator_version": self.evaluator_version,
            "prerequisites": list(self.prerequisites),
            "instructions_ref": {"sha256": self.instructions_ref.sha256, "size_bytes": self.instructions_ref.size_bytes},
            "completion_condition": self.completion_condition,
        }

    def manifest_id(self) -> str:
        return content_id(self.as_record())


@dataclass(frozen=True)
class DonorSelection:
    """P*(S1) = best_before(S1) under a declared comparator and tie rule."""

    s1_checkpoint_id: str
    donor_checkpoint_id: str | None
    comparator: str
    tie_rule: str
    candidate_checkpoint_ids: tuple[str, ...]
    prefix_cutoff_seq: int
    reason: str
    policy_version: str = "best-before-same-scope.v2"

    def as_record(self) -> dict[str, Any]:
        return dict(self.__dict__, candidate_checkpoint_ids=list(self.candidate_checkpoint_ids))


@dataclass(frozen=True)
class BranchSpec:
    """Inputs common to all branches from one S1."""

    run_id: str
    s1: Checkpoint
    donor: DonorSelection
    target: RecoveryTarget
    worker_model: str
    model_settings: Mapping[str, Any]
    generated_token_limit: int
    selection_kind: str  # primary_s1 / engineering_checkpoint / secondary
    runtime_policy: str
    packets: FrozenPackets | None = None
    recovery_settings: Mapping[str, Any] | None = None
    mask_spec: Mapping[str, Any] = field(default_factory=lambda: {
        "status": "unavailable", "reason": "No explicit source-bound observation mask was supplied"})

    def __post_init__(self) -> None:
        if self.runtime_policy == "fixed_from_donor":  # earlier name used by campaign.py
            object.__setattr__(self, "runtime_policy", "restore_captured")
        if self.runtime_policy not in RUNTIME_POLICIES:
            raise ValueError(f"unknown runtime policy {self.runtime_policy}")
        if self.selection_kind not in {"primary_s1", "engineering_checkpoint", "secondary"}:
            raise ValueError(f"unknown selection kind {self.selection_kind}")
        if self.s1.checkpoint_id != self.donor.s1_checkpoint_id:
            raise ValueError("donor selection is for a different S1")
        if not isinstance(self.mask_spec, Mapping):
            raise ValueError("mask_spec must be an explicit mask record or an unavailable record")
        if self.recovery_settings is not None and not recovery_settings_known(self.recovery_settings):
            raise ValueError("recovery_settings must declare call/format guards, request policy, tool timeout and detector settings")


@dataclass(frozen=True)
class ArmDef:
    arm_id: str
    workspace: str  # "s1" | "donor"
    history_policy: str
    packet_kind: str  # none | summary | t_typed | t_prose | t_view:<filter>
    study_cells: tuple[str, ...]
    intended_mismatches: tuple[str, ...] = ()


# Core arms. Identical conditions across studies are one arm with several cells.
CORE_ARMS: tuple[ArmDef, ...] = (
    ArmDef("continue", "s1", "native", "none", ("A:continue",)),
    ArmDef("reset_history", "s1", "fresh", "none", ("A:reset_history", "B:none", "2x2:S1_noT")),
    ArmDef("rollback_later_history", "donor", "native", "none", ("A:rollback_later_history",),
           ("history_mentions_absent_implementation",)),
    ArmDef("rollback_reset", "donor", "fresh", "none", ("A:rollback_reset", "2x2:D_noT")),
    ArmDef("o_masked", "s1", "o_masked", "none", ("A:o_masked",)),
    ArmDef("summary", "s1", "fresh", "summary", ("B:summary",)),
    ArmDef("t_prose", "s1", "fresh", "t_prose", ("B:t_prose",)),
    ArmDef("t_typed", "s1", "fresh", "t_typed", ("B:t_typed", "2x2:S1_T")),
    ArmDef("t_facts_only", "s1", "fresh", "t_view:facts_only", ("B:t_view:facts_only",)),
    ArmDef("t_facts_lessons", "s1", "fresh", "t_view:facts_lessons", ("B:t_view:facts_lessons",)),
    ArmDef("recomposition", "donor", "fresh", "t_typed", ("2x2:D_T",),
           ("t_observations_refer_to_S1_workspace",)),
)

# Explicit policy comparisons do not enter native W/H/O or W×T factor cells.
BASELINE_ARMS: tuple[ArmDef, ...] = (
    ArmDef("agentrewind", "s1", "native", "none", ("baseline:agentrewind",)),
)
ALL_ARMS = CORE_ARMS + BASELINE_ARMS


def resolve_arms(arm_ids: Sequence[str]) -> tuple[ArmDef, ...]:
    """Resolve an explicit configured lineup, preserving order; no implicit baseline."""
    registry = {a.arm_id: a for a in ALL_ARMS}
    if len(set(arm_ids)) != len(arm_ids):
        raise TreatmentError("duplicate arm IDs")
    unknown = [a for a in arm_ids if a not in registry]
    if unknown:
        raise TreatmentError(f"unknown arms: {unknown}")
    return tuple(registry[a] for a in arm_ids)


class TreatmentError(ValueError):
    pass


def _packet(spec: BranchSpec, kind: str) -> tuple[tuple[ArtifactRef, ...], dict[str, Any]]:
    if kind == "none":
        return (), {"kind": "none"}
    if spec.packets is None:
        raise TreatmentError(f"arm needs packet {kind} but no frozen packets were supplied")
    p = spec.packets
    if p.run_id != spec.s1.run_id or p.checkpoint_id != spec.s1.checkpoint_id:
        raise TreatmentError("frozen packets belong to a different source checkpoint/run")
    measure = p.token_measurements.get(kind)
    measurement_meta = {"freeze_id": p.freeze_id, "token_count": p.token_counts.get(kind),
                        "token_measurement": measure, "matching_report": dict(p.matching_report)}
    if measure is None or measure.get("fidelity") == "approximate_proxy":
        measurement_meta["approx_token_count"] = p.token_counts.get(kind)
    if kind == "summary":
        if p.summary_ref is None:
            raise TreatmentError("summary packet unavailable (failed or not generated)")
        return (p.summary_ref,), {"kind": "summary", "packet_id": p.summary_id, **measurement_meta}
    if kind not in p.packets:
        raise TreatmentError(f"frozen packets lack {kind}")
    pid, ref = p.packets[kind]
    return (ref,), {"kind": kind, "packet_id": pid, "source_version_id": p.version_id, **measurement_meta}


def build_manifest(spec: BranchSpec, arm: ArmDef, repeat_id: str) -> TreatmentManifest:
    if arm.history_policy not in HISTORY_POLICIES:
        raise TreatmentError(f"unknown history policy {arm.history_policy}")
    if arm.workspace == "donor":
        if spec.donor.donor_checkpoint_id is None:
            raise TreatmentError(f"arm {arm.arm_id} needs a donor but none is valid: {spec.donor.reason}")
        ws = spec.donor.donor_checkpoint_id
    else:
        ws = spec.s1.checkpoint_id
    history = spec.s1.checkpoint_id if arm.history_policy in {"native", "o_masked"} else "fresh"
    packet_refs, packet_meta = _packet(spec, arm.packet_kind)
    if packet_refs and arm.history_policy != "fresh":
        raise TreatmentError("packets are injected only into fresh-scaffold branches")
    donors = {"workspace": ws, "history": history}
    if spec.runtime_policy == "restore_captured":
        donors["runtime"] = ws  # the runner dereferences any runtime key, so omit it otherwise
    mismatches = list(arm.intended_mismatches)
    if arm.workspace == "donor" and spec.runtime_policy == "restore_captured":
        mismatches.append("W+R_restored_from_donor")
    metadata: dict[str, Any] = {
        "arm_id": arm.arm_id, "study_cells": list(arm.study_cells), "packet": packet_meta,
        "orientation_text": ORIENTATION_TEXT if arm.history_policy == "fresh" else None,
        "recovery_target": spec.target.as_record(), "recovery_target_manifest_id": spec.target.manifest_id(),
        "donor_selection": spec.donor.as_record(), "model_settings": dict(spec.model_settings),
        "packet_position": "after_target_and_orientation (provisional, unratified)",
        "isolation": "fresh_isolated_execution; no access to research store, other branches, grades or T",
    }
    if arm.history_policy == "o_masked":
        if spec.mask_spec.get("status") != "available":
            raise TreatmentError("observation mask unavailable: " + str(spec.mask_spec.get("reason", "missing bound policy")))
        if (spec.mask_spec.get("source_checkpoint_id") != spec.s1.checkpoint_id
                or spec.mask_spec.get("source_run_id") != spec.s1.run_id
                or spec.mask_spec.get("selection_kind") != spec.selection_kind
                or spec.mask_spec.get("placeholder") != MASK_PLACEHOLDER):
            raise TreatmentError("observation mask does not bind this source selection and exact placeholder")
        metadata["mask"] = dict(spec.mask_spec)
        metadata["mask_comparison_role"] = spec.mask_spec.get("comparison_role")
        if spec.mask_spec.get("comparison_role") == "secondary":
            metadata["study_cells"] = ["secondary:O_all_history"]
    if spec.recovery_settings is not None:
        metadata["recovery_settings"] = dict(spec.recovery_settings)
    if arm.arm_id == "agentrewind":
        from copy import deepcopy
        from ..baselines.agentrewind import POLICY
        if arm.workspace != "s1" or arm.history_policy != "native" or arm.packet_kind != "none":
            raise TreatmentError("AgentRewind baseline starts at aligned source W/R/H with no T packet")
        metadata["baseline_policy"] = deepcopy(POLICY)
        metadata["comparison_family"] = "agent_direct_policy_baseline; not a native causal-factor cell"
    treatment_id = content_id({"revision": "treatment-identity.v2", "run": spec.run_id,
        "s1": spec.s1.checkpoint_id, "arm": arm.arm_id, "repeat": repeat_id,
        "target": spec.target.manifest_id(), "packets": [r.sha256 for r in packet_refs],
        "donors": donors, "runtime_policy": spec.runtime_policy, "history_policy": arm.history_policy,
        "model": spec.worker_model, "model_settings": dict(spec.model_settings),
        "allowance": spec.generated_token_limit, "selection_kind": spec.selection_kind,
        "metadata": metadata})
    return TreatmentManifest(
        treatment_id=treatment_id, source_checkpoint_id=spec.s1.checkpoint_id, target_id=spec.target.target_id,
        stage_id=spec.target.stage_id, selection_kind=spec.selection_kind, component_donors=donors,
        history_policy=arm.history_policy, runtime_policy=spec.runtime_policy, worker_model=spec.worker_model,
        generated_token_limit=spec.generated_token_limit, repeat_id=repeat_id, packet_refs=packet_refs,
        source_scope={**({"session_scope": "full_trajectory", "include_returned_reasoning": False}
                         if spec.packets is None else spec.packets.binding.get("source_visibility", {}).get(
                             "scope", {"session_scope": "full_trajectory", "include_returned_reasoning": False})),
                      "through_checkpoint_id": spec.s1.checkpoint_id},
        intended_mismatches=tuple(mismatches), metadata=metadata,
    )


def build_all(spec: BranchSpec, repeats: Sequence[str], arms: Sequence[ArmDef] = CORE_ARMS
              ) -> tuple[list[TreatmentManifest], list[dict[str, Any]]]:
    """Return (valid manifests, skipped arms with reasons). Skips are recorded, never hidden."""
    manifests, skipped = [], []
    for arm in arms:
        for rep in repeats:
            try:
                manifests.append(build_manifest(spec, arm, rep))
            except TreatmentError as exc:
                skipped.append({"arm_id": arm.arm_id, "repeat_id": rep, "reason": str(exc)})
    return manifests, skipped


def validate_manifest(m: TreatmentManifest, store: ArtifactStoreProtocol, source_run_id: str) -> list[str]:
    if "manual_experiment" in m.metadata:
        from .manual import validate_manual_manifest
        problems = validate_manual_manifest(m, store)
        if problems:
            return problems
        if store.get_checkpoint(m.source_checkpoint_id).run_id != source_run_id:
            problems.append("source checkpoint belongs to a different run")
        return problems
    return _validate_manifest_common(m, store, source_run_id)


def _validate_manifest_common(m: TreatmentManifest, store: ArtifactStoreProtocol, source_run_id: str) -> list[str]:
    """Structural validation against the store: donors exist, are branchable, belong to the run and precede S1."""
    problems: list[str] = []
    manual = "manual_experiment" in m.metadata
    if "recovery_settings" in m.metadata and not recovery_settings_known(m.metadata["recovery_settings"]):
        problems.append("declared recovery settings lack a valid call cap, tool timeout or detector settings")
    try:
        s1 = store.get_checkpoint(m.source_checkpoint_id)
    except KeyError:
        return [f"source checkpoint {m.source_checkpoint_id} missing"]
    if s1.run_id != source_run_id:
        problems.append("source checkpoint belongs to a different run")
    if not s1.branchable:
        problems.append("source checkpoint is not branchable")
    if m.history_policy not in HISTORY_POLICIES or m.runtime_policy not in RUNTIME_POLICIES:
        problems.append("unknown history/runtime policy")
    if "workspace" not in m.component_donors:
        problems.append("workspace donor missing")
    expected_h = "fresh" if m.history_policy == "fresh" else s1.checkpoint_id
    if not manual and m.component_donors.get("history") != expected_h:
        problems.append("primary history must be source H or the declared fresh scaffold")
    if (m.target_id, m.stage_id) != (s1.scope.target_id, s1.scope.stage_id):
        problems.append("recovery target is not the source-stage fixed target")
    target = m.metadata.get("recovery_target", {})
    if target.get("requirement_version") != s1.scope.requirement_version:
        problems.append("recovery requirement binding differs from source")
    if (target.get("target_id"), target.get("stage_id"), target.get("evaluator_version")) != (
            s1.scope.target_id, s1.scope.stage_id, s1.scope.evaluator_version):
        problems.append("recovery target metadata differs from source scope")
    if m.metadata.get("recovery_target_manifest_id") != content_id(target):
        problems.append("recovery target manifest digest mismatch")
    if s1.evaluation_ref:
        source_grade = store.get_json(s1.evaluation_ref)
        if target.get("evaluator_version") != source_grade.get("evaluator_version"):
            problems.append("recovery evaluator differs from source grade")
    for component in ("workspace", "history", "runtime"):
        cid = m.component_donors.get(component)
        if cid in (None, "fresh", "n/a"):
            continue
        try:
            donor = store.get_checkpoint(cid)
        except KeyError:
            problems.append(f"{component} donor {cid} missing")
            continue
        if donor.run_id != s1.run_id:
            problems.append(f"{component} donor from another run/branch")
        if donor.event_sequence > s1.event_sequence or donor.ordinal > s1.ordinal:
            problems.append(f"{component} donor is after S1 (future leak)")
        if not donor.branchable:
            problems.append(f"{component} donor not branchable")
        if donor.scope.target_id != s1.scope.target_id:
            problems.append(f"{component} donor has a different target; scores not comparable")
        if (donor.scope.task_id, donor.scope.requirement_version, donor.scope.evaluator_version) != (
                s1.scope.task_id, s1.scope.requirement_version, s1.scope.evaluator_version):
            problems.append(f"{component} donor has incompatible task/requirement/evaluator bindings")
        if component == "runtime" and (donor.state.runtime_class.value != "R_CAPTURED" or donor.state.runtime is None):
            problems.append("runtime donor lacks captured runtime")
    if m.runtime_policy == "restore_captured" and s1.state.runtime_class.value != "R_CAPTURED":
        problems.append("runtime restore_captured requested but S1 runtime is not R_CAPTURED")
    if m.runtime_policy != "restore_captured" and "runtime" in m.component_donors:
        problems.append("runtime donor given without restore_captured policy")
    if m.runtime_policy == "restore_captured" and "runtime" not in m.component_donors:
        problems.append("runtime donor missing for captured runtime")
    if m.runtime_policy == "irrelevant" and s1.state.runtime_class.value != "R_IRRELEVANT":
        problems.append("runtime irrelevance differs from source capability")
    if m.runtime_policy == "unsupported":
        problems.append("required runtime restoration is unsupported")
    if m.runtime_policy == "discard_captured":
        # Runtime ablation: W (and H) restored, the captured R deliberately not restored (fresh container
        # runtime). A declared W+R bundle difference (Q15), never a W-only claim.
        if s1.state.runtime_class.value != "R_CAPTURED":
            problems.append("runtime discard_captured requires an R_CAPTURED decision checkpoint")
        if "runtime_discarded" not in tuple(m.intended_mismatches or ()):
            problems.append("runtime discard_captured must be declared as intended mismatch 'runtime_discarded'")
    for ref in m.packet_refs:
        try:
            store.get_bytes(ref)
        except Exception:
            problems.append(f"packet bytes {ref.sha256[:12]} unavailable")
    packet = m.metadata.get("packet", {})
    if (packet.get("kind", "none") == "none") != (not m.packet_refs):
        problems.append("packet kind and injected packet presence disagree")
    packet_source_id = m.metadata.get("packet_source_checkpoint_id", s1.checkpoint_id) if manual else s1.checkpoint_id
    if m.source_scope.get("through_checkpoint_id") != packet_source_id:
        problems.append("declared source scope does not stop at the source checkpoint")
    if m.packet_refs and packet.get("kind") == "research_packet":
        # Registered external packet: exact bytes, run/checkpoint binding, label and scope (research_packets.py).
        from .research_packets import research_packet_problems
        if not manual:
            problems.append("research packets enter only through explicit manual declarations")
        problems.extend(research_packet_problems(store, packet, tuple(m.packet_refs), m.source_scope, s1.run_id,
                                                 packet_source_id, s1.event_sequence))
    elif m.packet_refs:
        kind = "summary" if packet.get("kind") == "summary" else "packet"
        try:
            record = store.get_record(kind, packet.get("packet_id"))
            source = record.get("target_checkpoint_id") if kind == "summary" else record.get("lineage", {}).get("target_checkpoint_id")
            if source != packet_source_id:
                problems.append("packet is from a different source checkpoint")
            if record.get("text_sha256") != m.packet_refs[0].sha256 or len(m.packet_refs) != 1:
                problems.append("injected bytes differ from the frozen packet")
            if kind == "packet" and record.get("source_version_id") != packet.get("source_version_id"):
                problems.append("packet T source version differs from treatment metadata")
            source_record = (record if kind == "summary" else
                             store.get_record("t_version", record.get("source_version_id")))
            source_scope = source_record.get("source_visibility", {}).get("scope", {})
            if (source_scope.get("session_scope") != m.source_scope.get("session_scope")
                    or source_scope.get("include_returned_reasoning", False) !=
                       m.source_scope.get("include_returned_reasoning", False)):
                problems.append("declared treatment source scope differs from frozen information")
        except (KeyError, TypeError):
            problems.append("packet provenance record missing")
    if m.history_policy == "fresh" and m.metadata.get("orientation_text") != ORIENTATION_TEXT:
        problems.append("fresh scaffold must carry the canonical orientation text")
    if m.history_policy == "o_masked" and not m.metadata.get("mask"):
        problems.append("o_masked branch lacks a mask specification")
    elif m.history_policy == "o_masked":
        mask_source = store.get_checkpoint(m.component_donors["history"]) if manual else s1
        problems.extend(validate_observation_mask(store, mask_source, m.metadata["mask"], selection_kind=m.selection_kind))
    if not manual and m.history_policy != "fresh" and m.packet_refs:
        problems.append("packet on a non-fresh branch")
    if m.metadata.get("arm_id") == "agentrewind":
        from ..baselines.agentrewind import POLICY
        if (m.history_policy != "native" or m.component_donors.get("workspace") != s1.checkpoint_id
                or m.packet_refs or m.metadata.get("packet", {}).get("kind") != "none"):
            problems.append("AgentRewind baseline must start at aligned source W/R/H without T")
        if m.runtime_policy == "restore_captured" and m.component_donors.get("runtime") != s1.checkpoint_id:
            problems.append("AgentRewind baseline runtime must start at source")
        if m.metadata.get("baseline_policy") != POLICY:
            problems.append("AgentRewind baseline policy binding differs from executable policy")
    return problems


def select_donor(store: ArtifactStoreProtocol, run_id: str, s1: Checkpoint, *, comparator: str = "score_max",
                 tie_rule: str = "earliest_checkpoint") -> DonorSelection:
    """Q3: first attainment of best eligible pre-S score, independent of P labels.

    Primary eligibility uses the same native detector/comparison scope. A
    cross-scope donor sweep requires a separate explicit comparability policy.
    The old latest-tie choice remains available as a named secondary policy;
    historical selection records are never rewritten.
    """
    if comparator != "score_max" or tie_rule not in {"latest_checkpoint", "earliest_checkpoint"}:
        raise ValueError("unsupported donor comparator or tie rule")
    if s1.run_id != run_id:
        raise ValueError("donor population run differs from source run")
    cps = sorted(store.list_checkpoints(run_id), key=lambda c: c.ordinal)  # type: ignore[attr-defined]
    candidates = []
    for c in cps:
        if c.ordinal >= s1.ordinal or c.event_sequence >= s1.event_sequence or c.checkpoint_id == s1.checkpoint_id:
            continue
        if not c.branchable or c.scope != s1.scope or c.evaluation_ref is None:
            continue
        ev = store.get_json(c.evaluation_ref)
        score = ev.get("score")
        if ev.get("valid") is not True or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            continue
        if (ev.get("task_id"), ev.get("stage_id"), ev.get("target_id"), ev.get("evaluator_version")) != (
                s1.scope.task_id, s1.scope.stage_id, s1.scope.target_id, s1.scope.evaluator_version):
            continue
        candidates.append((float(score), c.ordinal, c))
    if not candidates:
        return DonorSelection(s1.checkpoint_id, None, comparator, tie_rule, (), s1.event_sequence,
                              "no valid graded branchable same-scope checkpoint strictly before S1")
    best_score = max(s for s, _, _ in candidates)
    tied = [c for s, _, c in candidates if s == best_score]
    chosen = tied[-1] if tie_rule == "latest_checkpoint" else tied[0]
    return DonorSelection(s1.checkpoint_id, chosen.checkpoint_id, comparator, tie_rule,
                          tuple(c.checkpoint_id for _, _, c in candidates), s1.event_sequence,
                          f"best score {best_score} among {len(candidates)} candidates; tie rule {tie_rule}")


def manifest_digest(m: TreatmentManifest) -> str:
    return content_id(canonical_json(m).decode("utf-8"))
