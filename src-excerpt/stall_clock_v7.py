"""Stall-clock detector (v7): no measurable progress for a sustained stretch of effort.

A stall warning (S) is emitted at a valid, branchable graded checkpoint when, within
the current scope (stage), the worker has spent at least ``stagnation_token_threshold``
generated tokens AND at least ``TURN_THRESHOLD`` model turns since the last measurable
progress, the scope is not already solved, and the episode is armed.

Measurable progress at a valid grade is any strict improvement over the best seen in
this scope of: completion score (by at least the task-declared ``score_min_gain``, default 0), number of passing checks, or number of checks that
execute (PASS or FAIL rather than ERROR/BLOCKED/NOT_RUN/INFRA). The scope baseline
grade initializes these bests and is not progress.

The clock is checked after every model turn, not only at edit-triggered grades. When it
is met and no grade happened during that turn, the detector requests a private grade;
the runner then captures a checkpoint, and S is decided on that fresh grade.

v8 (``stall-clock/v8``) keeps all of the above and adds one more way to warn, for tasks
whose native protocol answers a submission with its first unmet requirement. When a
submission is rejected on the same requirement as the previous submission in this scope,
with no measurable progress in between, the detector requests a private grade; the runner
takes it after the feedback reaches the worker (the submission checkpoint itself ends in the
exit marker and is not branchable), and S is decided on that grade without the token or turn
threshold. Validity, branchability, not-solved and one-warning-per-episode still apply. v7
behaviour and records are unchanged.

Deliberately absent (see docs/research/detector-v7/README.md): behavioral recurrence
gates, edit-to-check relevance maps, failure-signature policies, frontier reachability
completeness and separate staged/single-stage thresholds. Hindsight review of 77 runs
showed those gates blocked most genuine stalls without improving outcome precision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STALL_CLOCK_RULE_VERSION = "stall-clock/v7"
STALL_CLOCK_V8_RULE_VERSION = "stall-clock/v8"
STALL_CLOCK_V81_RULE_VERSION = "stall-clock/v8.1"
STALL_CLOCK_RULE_VERSIONS = (STALL_CLOCK_RULE_VERSION, STALL_CLOCK_V8_RULE_VERSION, STALL_CLOCK_V81_RULE_VERSION)
SUBMISSION_VERDICT_RULE_VERSIONS = (STALL_CLOCK_V8_RULE_VERSION, STALL_CLOCK_V81_RULE_VERSION)
TURN_THRESHOLD = 5


@dataclass
class _Scope:
    scope_id: str
    stage_id: str | None
    anchor_tokens: int
    baseline_done: bool = False
    best_score: float | None = None
    best_pass: int = -1
    best_exec: int = -1
    solved: bool = False
    last_valid: bool = False
    turns: int = 0
    armed: bool = True
    open_stall: dict[str, Any] | None = None
    graded_this_turn: bool = False
    next_request_turn: int = TURN_THRESHOLD
    progress_count: int = 0
    last_graded_checkpoint: str | None = None
    last_rejection: tuple[str, int] | None = None  # (criterion, progress_count) of the latest rejected submission
    pending_repeat: str | None = None  # v8: repeated-rejection criterion awaiting its post-feedback grade


class StallClockDetector:
    """Online detector. Feed input records in strictly increasing seq order."""

    def __init__(self, config: Any, *, run_id: str) -> None:
        if config.rule_version not in STALL_CLOCK_RULE_VERSIONS:
            raise ValueError("StallClockDetector requires a stall-clock rule version")
        self.config = config
        self.version = config.rule_version
        self.threshold = int(config.stagnation_token_threshold)
        self.run_id = run_id
        self._last_seq: int | None = None
        self.known_tokens = 0
        self.unknown_calls = 0
        self.branchable: dict[str, bool] = {}
        self.counters = {"P": 0, "F": 0, "S": 0, "N": 0}
        self.scope: _Scope | None = None

    def feed(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        seq = item["seq"]
        if self._last_seq is not None and seq <= self._last_seq:
            raise ValueError(f"detector inputs must be strictly ordered: {seq} after {self._last_seq}")
        self._last_seq = seq
        handler = getattr(self, f"_on_{item['type']}", None)
        if handler is None:
            raise ValueError(f"unknown detector input type {item['type']!r}")
        return handler(item)

    # ------------------------------------------------------------------ inputs
    def _on_usage(self, u: dict[str, Any]) -> list[dict[str, Any]]:
        tokens = u["generated_tokens"]
        if tokens is None:
            self.unknown_calls += 1
        elif tokens < 0:
            raise ValueError("generated_tokens cannot be negative")
        else:
            self.known_tokens += tokens
        return []

    def _on_checkpoint(self, c: dict[str, Any]) -> list[dict[str, Any]]:
        self.branchable[c["checkpoint_id"]] = bool(c["valid"])
        return []

    def _on_mutation(self, m: dict[str, Any]) -> list[dict[str, Any]]:
        return []  # Edits matter only through the grades they produce.

    def _on_scope_start(self, s: dict[str, Any]) -> list[dict[str, Any]]:
        out = self._close_scope(s["seq"], "scope_closed")
        previous = self.scope.scope_id if self.scope else None
        self.scope = _Scope(s["scope_id"], s.get("stage_id"), self.known_tokens)
        out.append({"record": "detector_scope", "event": "scope_initialized", "seq": s["seq"],
                    "scope_id": s["scope_id"], "stage_id": s.get("stage_id"), "previous_scope_id": previous,
                    "reason": s.get("reason"), "rule_version": self.version,
                    "note": "Scope start resets the stall clock; it is not progress or escape."})
        return out

    def _on_terminal(self, t: dict[str, Any]) -> list[dict[str, Any]]:
        return self._close_scope(t["seq"], f"trajectory_terminal:{t.get('reason')}")

    def _on_turn_boundary(self, t: dict[str, Any]) -> list[dict[str, Any]]:
        sc = self._scope(t)
        sc.turns += 1
        graded, sc.graded_this_turn = sc.graded_this_turn, False
        if graded or not sc.baseline_done or not self._clock_met(sc):
            return []
        if not sc.armed or sc.solved or sc.turns < sc.next_request_turn:
            return []
        sc.next_request_turn = sc.turns + TURN_THRESHOLD
        return [{"record": "grade_request", "seq": t["seq"], "scope_id": sc.scope_id,
                 "rule_version": self.version, "reason": "stall_clock",
                 "stagnation": self._stagnation(sc),
                 "note": "Private grade at a completed turn boundary; not an S and not worker feedback."}]

    def _on_grade(self, g: dict[str, Any]) -> list[dict[str, Any]]:
        sc = self._scope(g)
        sc.graded_this_turn = True
        cp = g["checkpoint_id"]
        sc.last_graded_checkpoint = cp
        base = {"record": "detector_decision", "seq": g["seq"], "checkpoint_id": cp, "scope_id": sc.scope_id,
                "stage_id": sc.stage_id, "rule_version": self.version,
                "score": g.get("score"), "passing_checks": g.get("passing_checks"),
                "executed_checks": g.get("executed_checks")}
        valid = bool(g["valid"]) and g.get("score") is not None
        sc.last_valid = valid
        if not valid:
            return [dict(base, decision="invalid_grade", stagnation=self._stagnation(sc),
                         note="Untrustworthy measurement: no progress, no S until a later valid grade.")]
        score, npass, nexec = float(g["score"]), int(g.get("passing_checks") or 0), int(g.get("executed_checks") or 0)
        if g.get("scope_baseline") or not sc.baseline_done:
            sc.baseline_done = True
            self._set_best(sc, score, npass, nexec)
            sc.solved = bool(g.get("full_solve"))
            self._reset_clock(sc)
            return [dict(base, decision="scope_baseline", best={"score": score, "passing": npass, "executed": nexec})]
        # The task category may declare the smallest score gain that counts as progress (measured
        # grading noise, e.g. time-limited optimization programs); default 0 = any strict gain.
        # A declared margin means the measurement is noisy, so per-check pass counts from that same
        # measurement are not progress either; more checks able to run (build/validity) still are.
        declared = float(getattr(self.config, "score_min_gain", 0.0) or 0.0)
        margin = max(declared, 1e-12)
        new_best = sc.best_score is None or score >= sc.best_score + margin
        more_pass, more_exec = npass > sc.best_pass and declared == 0.0, nexec > sc.best_exec
        out: list[dict[str, Any]] = []
        if new_best or more_pass or more_exec:
            stagnation = self._stagnation(sc)
            if new_best:
                out.append(self._annotation("P", cp, g["seq"], sc, score=score, previous_best=sc.best_score))
            if more_pass or more_exec:
                out.append(self._annotation("F", cp, g["seq"], sc, passing_checks=npass, executed_checks=nexec,
                                            previous_passing=sc.best_pass, previous_executed=sc.best_exec))
            if sc.open_stall is not None:
                s = sc.open_stall
                out.append(self._annotation("N", cp, g["seq"], sc, linked_s_annotation_id=s["annotation_id"],
                                            tokens_since_s=self.known_tokens - s["known_tokens"],
                                            turns_since_s=sc.turns - s["turns"]))
                sc.open_stall = None
            self._set_best(sc, score, npass, nexec)
            sc.solved = sc.solved or bool(g.get("full_solve"))
            sc.progress_count += 1
            sc.pending_repeat = None
            self._reset_clock(sc)
            out.insert(0, dict(base, decision="progress", new_best=new_best, more_passing=more_pass,
                               more_executing=more_exec, stagnation_before=stagnation))
            return out
        sc.solved = sc.solved or bool(g.get("full_solve"))
        repeat, sc.pending_repeat = sc.pending_repeat, None
        conditions = {"valid_grade": True, "not_solved": not sc.solved, "episode_armed": sc.armed,
                      "token_threshold_met": self.known_tokens - sc.anchor_tokens >= self.threshold,
                      "turn_threshold_met": sc.turns >= TURN_THRESHOLD,
                      "branchable_checkpoint": self.branchable.get(cp, False)}
        if repeat is not None:  # v8 repeated rejection: the clock thresholds do not apply
            conditions = {k: ok for k, ok in conditions.items() if not k.endswith("_threshold_met")}
        failed = [k for k, v in conditions.items() if not v]
        decision = dict(base, eligibility=conditions, failed_eligibility_conditions=failed,
                        stagnation=self._stagnation(sc))
        if self.version in SUBMISSION_VERDICT_RULE_VERSIONS:
            decision["trigger"] = "repeated_rejection" if repeat is not None else "stall_clock"
        if failed:
            return [dict(decision, decision="no_warning")]
        if self.version != STALL_CLOCK_V81_RULE_VERSION:
            sc.armed = False  # v7/v8: one warning per no-progress episode
        trigger = {} if self.version not in SUBMISSION_VERDICT_RULE_VERSIONS else (
            {"trigger": "repeated_rejection", "criterion_id": repeat} if repeat is not None else {"trigger": "stall_clock"})
        ann = self._annotation("S", cp, g["seq"], sc, stagnation=self._stagnation(sc), **trigger)
        if sc.open_stall is None:  # N/followup stay linked to the first S of the stretch (v8.1 may add more)
            sc.open_stall = {"annotation_id": ann["annotation_id"], "known_tokens": self.known_tokens, "turns": sc.turns}
        return [dict(decision, decision="stall_warning", annotation_id=ann["annotation_id"]), ann]

    def _on_submission_verdict(self, v: dict[str, Any]) -> list[dict[str, Any]]:
        """v8: the native verdict for the submission graded at ``checkpoint_id``, fed after its feedback."""
        if self.version not in SUBMISSION_VERDICT_RULE_VERSIONS:
            raise ValueError("submission_verdict inputs require stall-clock/v8 or v8.1")
        sc = self._scope(v)
        criterion = v.get("criterion_id")
        if v["status"] == "pass" or criterion is None:
            sc.last_rejection = None
            return []
        previous, sc.last_rejection = sc.last_rejection, (criterion, sc.progress_count)
        if previous != (criterion, sc.progress_count) or not sc.armed or sc.solved:
            return []  # first rejection on this requirement, progress since the previous one, or already warned
        sc.pending_repeat = criterion
        return [{"record": "grade_request", "seq": v["seq"], "scope_id": sc.scope_id, "rule_version": self.version,
                 "reason": "repeated_rejection", "criterion_id": criterion,
                 "submission_checkpoint_id": v["checkpoint_id"], "stagnation": self._stagnation(sc),
                 "note": "Private grade after the rejection feedback; not an S and not worker feedback."}]

    # ----------------------------------------------------------------- helpers
    def _scope(self, item: dict[str, Any]) -> _Scope:
        sc = self.scope
        if sc is None or item.get("scope_id") != sc.scope_id:
            raise ValueError(f"{item['type']} received outside its scope")
        return sc

    def _clock_met(self, sc: _Scope) -> bool:
        return self.known_tokens - sc.anchor_tokens >= self.threshold and sc.turns >= TURN_THRESHOLD

    def _stagnation(self, sc: _Scope) -> dict[str, Any]:
        return {"tokens_since_progress": self.known_tokens - sc.anchor_tokens, "turns_since_progress": sc.turns,
                "token_threshold": self.threshold, "turn_threshold": TURN_THRESHOLD,
                "unknown_usage_calls": self.unknown_calls}

    @staticmethod
    def _set_best(sc: _Scope, score: float, npass: int, nexec: int) -> None:
        sc.best_score = score if sc.best_score is None else max(sc.best_score, score)
        sc.best_pass, sc.best_exec = max(sc.best_pass, npass), max(sc.best_exec, nexec)

    def _reset_clock(self, sc: _Scope) -> None:
        sc.anchor_tokens, sc.turns, sc.armed = self.known_tokens, 0, True
        sc.next_request_turn = TURN_THRESHOLD

    def _annotation(self, label: str, checkpoint_id: str, seq: int, sc: _Scope, **details: Any) -> dict[str, Any]:
        self.counters[label] += 1
        return {"record": "annotation", "annotation_id": f"{self.run_id}/{label}{self.counters[label]}",
                "label": label, "label_index": self.counters[label], "target_checkpoint_id": checkpoint_id,
                "assigned_at_seq": seq, "evidence_cutoff_seq": seq, "scope_id": sc.scope_id,
                "stage_id": sc.stage_id, "rule_version": self.version,
                "known_tokens": self.known_tokens, **details}

    def _close_scope(self, seq: int, reason: str) -> list[dict[str, Any]]:
        sc = self.scope
        if sc is None or sc.open_stall is None:
            return []
        s, sc.open_stall = sc.open_stall, None
        return [{"record": "followup", "seq": seq, "s_annotation_id": s["annotation_id"], "scope_id": sc.scope_id,
                 "reason": reason, "escaped": False, "rule_version": self.version,
                 "tokens_observed_since_s": self.known_tokens - s["known_tokens"],
                 "turns_observed_since_s": sc.turns - s["turns"],
                 "note": "No natural escape observed before the scope closed; censored, not proven failure."}]
