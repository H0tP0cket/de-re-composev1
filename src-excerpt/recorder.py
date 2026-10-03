"""Event emission with the execution/observer conventions.

Kinds and visibility (confirmed with the observer and integration sessions):

| kind              | visibility       | refs |
| ----------------- | ---------------- | ---- |
| task_disclosure   | WORKER_VISIBLE   | delivered_ref = exact delivered text |
| model_request     | WORKER_VISIBLE   | input_ref = exact request JSON bytes |
| model_attempt     | RESEARCH_PRIVATE | input_ref = exact posted bytes; output_ref = raw provider bytes; usage and transport metadata |
| model_response    | WORKER_VISIBLE   | output_ref = normalized accepted message; usage |
| format_error      | WORKER_VISIBLE   | delivered_ref = exact error message delivered |
| tool_call         | WORKER_VISIBLE   | canonical payload.arguments; input_ref = original provider argument text when available |
| tool_result       | RESEARCH_PRIVATE | output_ref = raw bytes; execution status and duration, before rendering |
| tool_render       | RESEARCH_PRIVATE | raw-result parent; rendered_ref; not proof of delivery |
| observation_delivery | WORKER_VISIBLE | delivered_ref = tool content actually appended to H |
| treatment_packet  | WORKER_VISIBLE   | delivered_ref = exact packet bytes |
| mutation, capture, grade, detector, annotation, lifecycle, terminal, incident | RESEARCH_PRIVATE | |

Sequence numbers are assigned by the store; a checkpoint's event_sequence is an
inclusive cut.
"""

from __future__ import annotations

from typing import Any

from ..contracts import ArtifactRef, ArtifactStoreProtocol, EventRecord, Scope, Usage, Visibility
from ..persistence import persist

WORKER_VISIBLE_KINDS = {"task_disclosure", "model_request", "model_response", "format_error", "tool_call",
                        "observation_delivery", "treatment_packet"}


class Recorder:
    def __init__(self, store: ArtifactStoreProtocol, run_id: str) -> None:
        self.store = store
        self.run_id = run_id
        self.scope: Scope | None = None
        self.last_sequence = 0
        self.persistence_failure: BaseException | None = None
        self.capture_failure: BaseException | None = None

    def persist(self, operation, *, evidence):
        try:
            return persist(operation, evidence=evidence)
        except BaseException as exc:
            self.persistence_failure = exc
            raise

    def put_bytes(self, data: bytes, **kwargs) -> ArtifactRef:
        return self.persist(lambda: self.store.put_bytes(data, **kwargs), evidence={"bytes": data})

    def put_json(self, value: Any, **kwargs) -> ArtifactRef:
        return self.persist(lambda: self.store.put_json(value, **kwargs), evidence=value)

    def text_ref(self, text: str, visibility: Visibility) -> ArtifactRef:
        return self.put_bytes(text.encode("utf-8"), media_type="text/plain; charset=utf-8", visibility=visibility)

    def emit(self, kind: str, *, payload: dict[str, Any] | None = None, actor: str = "harness",
             call_id: str | None = None, attempt: int | None = None, parent_event_id: str | None = None,
             input_ref: ArtifactRef | None = None, output_ref: ArtifactRef | None = None,
             delivered_ref: ArtifactRef | None = None, usage: Usage | None = None,
             visibility: Visibility | None = None) -> EventRecord:
        if visibility is None:
            visibility = Visibility.WORKER_VISIBLE if kind in WORKER_VISIBLE_KINDS else Visibility.RESEARCH_PRIVATE
        event = EventRecord(run_id=self.run_id, kind=kind, actor=actor, scope=self.scope, parent_event_id=parent_event_id,
                            call_id=call_id, attempt=attempt, visibility=visibility, input_ref=input_ref,
                            output_ref=output_ref, delivered_ref=delivered_ref, usage=usage, payload=payload or {})
        saved = self.persist(lambda: self.store.append_event(event), evidence=event)
        self.last_sequence = saved.sequence
        return saved

    def incident(self, what: str, **details: Any) -> EventRecord:
        return self.emit("incident", payload={"what": what, **details})
