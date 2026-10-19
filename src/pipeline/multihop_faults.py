"""Fault injection and observation-only diagnosis for the MultiHop path.

Each layer is defined as a consistency violation between a recorded layer input and
the upstream output it should have received. The injected label lives only in
TrialRecord (evaluator side); diagnose() receives a MultiHopObservation only.
Synthetic-fixture code (src/pipeline/agent_runner.py) is untouched.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from src.schema import FaultType

N_SPECIALISTS = 3
NOVELTY_THRESHOLD = 0.5
ABSTENTION = re.compile(
    r"does not contain|do not contain|insufficient|no evidence|not enough|"
    r"cannot determine|unable to determine|no relevant|^\s*none\.?\s*$",
    re.IGNORECASE,
)
STOP = frozenset(
    "that with this from have has had are was were will would could should their "
    "there been being into about which while when what who than then them they "
    "your not but also more most such only over some any all can may its our out "
    "said says report facts key".split()
)


class SpecialistObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_id: str
    retrieved_doc_ids: List[str]
    input_docs: List[Dict[str, str]]
    output: str


class MultiHopObservation(BaseModel):
    """Everything the diagnoser may read. extra='forbid' blocks added label fields."""
    model_config = ConfigDict(extra="forbid")
    query: str
    specialists: List[SpecialistObservation]
    arbiter_inputs: List[Dict[str, str]]
    final_verdict: str


@dataclass(frozen=True)
class InjectionRecord:
    """Evaluator-only record of what was injected."""
    layer: FaultType
    target_agent: Optional[str] = None
    transformation: str = "none"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialRecord:
    observation: MultiHopObservation
    injected_layer: FaultType
    injection: InjectionRecord
    predicted_layer: FaultType
    prediction_made_before_label_attached: bool = True


# ----------------------------------------------------------------- helpers
def _pick(qid: str, condition: str, salt: str, n: int) -> int:
    digest = hashlib.sha256(f"{qid}|{condition}|{salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z]{4,}", text.lower()) if t not in STOP}


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return parts[0]


def assign_chunks(docs: List[dict]) -> List[List[dict]]:
    """Round-robin assignment; every document goes to exactly one specialist."""
    return [[d for pos, d in enumerate(docs) if pos % N_SPECIALISTS == i]
            for i in range(N_SPECIALISTS)]


# --------------------------------------------------------------- injectors
def apply_retrieval_fault(chunks, pool, qid, condition):
    """Replace one required claim document by a neutral pool filler (count preserved)."""
    candidates = [i for i, c in enumerate(chunks) if any(d.get("role") == "claim" for d in c)]
    if not candidates:
        raise ValueError("No claim document available for retrieval-fault injection")
    agent = candidates[_pick(qid, condition, "retrieval", len(candidates))]
    in_cell_ids = {d["doc_id"] for c in chunks for d in c}
    in_cell_hashes = {d["content_sha256"] for c in chunks for d in c}
    target = next(d for d in chunks[agent] if d.get("role") == "claim")
    fresh = [p for p in pool if p["content_sha256"] not in in_cell_hashes
             and p["doc_id"] not in in_cell_ids]
    same_unit = [p for p in pool if p["slot_unit_id"] == target["slot_unit_id"]]
    replacement = (fresh or same_unit or pool)[0]
    actual = [list(c) for c in chunks]
    actual[agent] = [replacement if d["doc_id"] == target["doc_id"] else d for d in actual[agent]]
    return actual, InjectionRecord(
        FaultType.RETRIEVAL, f"Specialist_{agent + 1}", "claim_document_replaced_by_filler",
        {"replaced_doc_id": target["doc_id"], "replacement_doc_id": replacement["doc_id"]})


def apply_specialist_fault(outputs, actual_chunks, pool, qid, condition):
    """Replace one specialist output by an ungrounded corpus sentence (wrong-claim report)."""
    agent = _pick(qid, condition, "specialist", len(outputs))
    seen = " ".join(d["content"] for d in actual_chunks[agent])
    options = [p for p in pool if p["content"] not in seen]
    chosen = (options or pool)[_pick(qid, condition, "specialist_doc", len(options or pool))]
    new = list(outputs)
    new[agent] = "Key facts: " + _first_sentence(chosen["content"])
    return new, InjectionRecord(
        FaultType.SPECIALIST, f"Specialist_{agent + 1}", "output_replaced_by_ungrounded_sentence",
        {"source_doc_id": chosen["doc_id"]})


def apply_arbiter_fault(outputs, qid, condition):
    """Deliver the arbiter all specialist reports except one (drop is recorded)."""
    live = [i for i, o in enumerate(outputs) if o.strip()] or list(range(len(outputs)))
    drop = live[_pick(qid, condition, "arbiter", len(live))]
    delivered = [{"agent_id": f"Specialist_{i + 1}", "output": o}
                 for i, o in enumerate(outputs) if i != drop]
    return delivered, InjectionRecord(
        FaultType.ARBITER, "Arbiter", "specialist_report_dropped",
        {"dropped_agent": f"Specialist_{drop + 1}"})


# ------------------------------------------------------------- observation
def build_observation(query, retrieved, actual, outputs, arbiter_inputs, final_verdict,
                      render=None):
    render = render or (lambda d: d["content"])
    specialists = [
        SpecialistObservation(
            agent_id=f"Specialist_{i + 1}",
            retrieved_doc_ids=[d["doc_id"] for d in retrieved[i]],
            input_docs=[{"doc_id": d["doc_id"], "content": render(d)} for d in actual[i]],
            output=outputs[i])
        for i in range(len(outputs))]
    return MultiHopObservation(query=query, specialists=specialists,
                               arbiter_inputs=arbiter_inputs, final_verdict=final_verdict)


# --------------------------------------------------------------- diagnoser
def _novelty(output: str, grounding: str) -> float:
    out = _tokens(output)
    return len(out - _tokens(grounding)) / len(out) if out else 0.0


def diagnose(observation: MultiHopObservation) -> FaultType:
    """Observation-only layer diagnosis; upstream layers are checked first."""
    for s in observation.specialists:
        if s.retrieved_doc_ids != [d["doc_id"] for d in s.input_docs]:
            return FaultType.RETRIEVAL
    for s in observation.specialists:
        if not s.output.strip() or ABSTENTION.search(s.output):
            continue
        grounding = observation.query + " " + " ".join(d["content"] for d in s.input_docs)
        if _novelty(s.output, grounding) > NOVELTY_THRESHOLD:
            return FaultType.SPECIALIST
    produced = [(s.agent_id, s.output) for s in observation.specialists]
    delivered = [(a["agent_id"], a["output"]) for a in observation.arbiter_inputs]
    if produced != delivered:
        return FaultType.ARBITER
    return FaultType.NONE


# ------------------------------------------------------------------- trial
def execute_fault_trial(question: dict, condition: str, layer: FaultType,
                        generate: Callable[[str, str, str], str],
                        apply_injection: bool = True,
                        exclude_doc_ids=(), render_doc=None) -> TrialRecord:
    """Run one trajectory. generate(role, context, question) -> text.

    apply_injection=False declares the label without mutating anything; it exists so
    tests can change ONLY the hidden label.
    exclude_doc_ids removes documents AFTER specialist assignment, so remaining
    documents keep their specialist (an intervention must not reshuffle the others).
    render_doc(doc) -> text is what the specialist (and the observation) sees.
    """
    qid, query = question["sample_id"], question["query"]
    docs = question["conditions"][condition]["documents"]
    pool = question["filler_pool"]
    excluded = set(exclude_doc_ids)
    render = render_doc or (lambda d: d["content"])
    retrieved = [[d for d in chunk if d["doc_id"] not in excluded]
                 for chunk in assign_chunks(docs)]
    actual, record = retrieved, None
    if apply_injection and layer == FaultType.RETRIEVAL:
        actual, record = apply_retrieval_fault(retrieved, pool, qid, condition)

    outputs = []
    for chunk in actual:
        if not chunk:
            outputs.append("Insufficient evidence: no documents provided.")
            continue

        context = "\n\n".join(render(d) for d in chunk)
        outputs.append(generate("specialist", context, query))
    if apply_injection and layer == FaultType.SPECIALIST:
        outputs, record = apply_specialist_fault(outputs, actual, pool, qid, condition)

    delivered = [{"agent_id": f"Specialist_{i + 1}", "output": o} for i, o in enumerate(outputs)]
    if apply_injection and layer == FaultType.ARBITER:
        delivered, record = apply_arbiter_fault(outputs, qid, condition)

    arbiter_context = "\n\n".join(f"{a['agent_id']} report:\n{a['output']}" for a in delivered)
    verdict = generate("arbiter", arbiter_context, query)
    observation = build_observation(query, retrieved, actual, outputs, delivered, verdict, render)
    predicted = diagnose(observation)  # prediction fixed before the label is attached
    return TrialRecord(observation=observation, injected_layer=layer,
                       injection=record or InjectionRecord(layer), predicted_layer=predicted)
