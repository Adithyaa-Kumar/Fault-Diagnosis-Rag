import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Union
from src.schema import (
    DiagnosticObservation, ExecutionTrace, SpecialistTrace, FaultType,
    ProvenanceDocument,
)
from src.evaluation.scoring import AnswerScorer


class BaselinePipeline:
    """Deterministic mock parallel fan-in RAG pipeline for Phase 0 plumbing tests."""
    def __init__(self, model_name: str = "deterministic-phase0-mock"):
        self.model_name = model_name
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def run_specialist(
        self, agent_id: str, assigned_docs: List[ProvenanceDocument],
        retrieved_docs: List[ProvenanceDocument], query: str,
        inject_specialist_fault: bool = False,
    ) -> SpecialistTrace:
        context = "\n".join(f"[{d.doc_id}]: {d.content}" for d in assigned_docs)
        actual_inputs = [f"Query: {query}", f"Context: {context}"]
        raw_output = f"Specialist evidence: {context}"
        # Do not expose an explicit '[INJECTED FAULT]' marker to the diagnostic model.
        if inject_specialist_fault:
            raw_output = "Specialist evidence: The Department of Treasury authorized the stabilization."
        return SpecialistTrace(
            agent_id=agent_id,
            assigned_docs=[d.doc_id for d in assigned_docs],
            retrieved_docs=list(retrieved_docs),
            actual_inputs=actual_inputs,
            raw_output=raw_output,
        )

    @staticmethod
    def _claim_supported(text: str, claim: dict) -> bool:
        terms = claim.get("match_terms", [])
        text_lower = text.lower()
        return bool(terms) and all(term.lower() in text_lower for term in terms)

    def run_arbiter(self, specialist_traces: List[SpecialistTrace], gold_answer: str,
                    required_claims: List[dict], inject_arbiter_fault: bool = False) -> str:
        if inject_arbiter_fault:
            return "Arbiter Verdict: The Department of Treasury authorized the action."
        combined_text = " ".join(t.raw_output for t in specialist_traces)
        if all(self._claim_supported(combined_text, claim) for claim in required_claims):
            return f"Arbiter Verdict: {gold_answer}"
        supported = [c["claim_text"] for c in required_claims if self._claim_supported(combined_text, c)]
        if supported:
            return "Arbiter Verdict: Partial evidence: " + "; ".join(supported) + ". Insufficient evidence to establish all required claims."
        return "Arbiter Verdict: Insufficient evidence to determine the required claims."

    @staticmethod
    def diagnose(observation: DiagnosticObservation) -> FaultType:
        """Heuristic diagnosis using only observable trace data, never injected labels."""
        # Retrieval corruption is visible when the actual specialist input document IDs
        # differ from the documents recorded as retrieved for that specialist.
        for specialist in observation.specialist_traces:
            retrieved_ids = {d.doc_id for d in specialist.retrieved_docs}
            assigned_ids = set(specialist.assigned_docs)
            if retrieved_ids != assigned_ids:
                return FaultType.RETRIEVAL

        # Detect an unsupported specialist assertion by comparing its output to its input.
        for specialist in observation.specialist_traces:
            output = specialist.raw_output.lower()
            if "department of treasury" in output:
                input_text = " ".join(specialist.actual_inputs).lower()
                if "department of treasury" not in input_text:
                    return FaultType.SPECIALIST

        # Arbiter anomaly is diagnosed from observable disagreement only.
        # Do not consult gold_answer, required_claims, or injected labels here.
        specialist_text = " ".join(s.raw_output for s in observation.specialist_traces).lower()
        verdict_text = observation.final_verdict.lower()
        specialists_support_toc = (
            "trade oversight commission" in specialist_text
            or " toc " in f" {specialist_text} "
        )
        arbiter_asserts_conflicting_agency = (
            "department of treasury" in verdict_text
            or "ministry of finance" in verdict_text
        )
        if specialists_support_toc and arbiter_asserts_conflicting_agency:
            return FaultType.ARBITER
        return FaultType.NONE

    def execute_pipeline(
        self, sample_path: str, condition_key: str, injected_layer: FaultType,
        documents_override: Optional[Sequence[Union[ProvenanceDocument, dict]]] = None,
        write_log: bool = True,
    ) -> ExecutionTrace:
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        query_id, query = data["query_id"], data["query"]
        gold_answer = data["gold_answer"]
        required_claims = data["required_claims"]
        raw_docs = documents_override if documents_override is not None else data["conditions"][condition_key]
        documents = [d if isinstance(d, ProvenanceDocument) else ProvenanceDocument(**d) for d in raw_docs]
        original_corpus = [ProvenanceDocument(**d) for d in data["conditions"][condition_key]]

        chunks = [[documents[i]] if i < len(documents) else [] for i in range(3)]
        specialist_traces: List[SpecialistTrace] = []
        for idx, original_chunk in enumerate(chunks):
            actual_chunk = list(original_chunk)
            if injected_layer == FaultType.RETRIEVAL and idx == 0:
                actual_chunk = [ProvenanceDocument(
                    doc_id="faulty_doc", content="Irrelevant background information.",
                    source_cluster_id="bad_cluster", claim_id="claim_irrelevant",
                )]
            trace = self.run_specialist(
                agent_id=f"Specialist_{idx + 1}",
                assigned_docs=actual_chunk,
                retrieved_docs=list(original_chunk),
                query=query,
                inject_specialist_fault=(injected_layer == FaultType.SPECIALIST and idx == 0),
            )
            specialist_traces.append(trace)

        final_verdict = self.run_arbiter(
            specialist_traces, gold_answer, required_claims,
            inject_arbiter_fault=(injected_layer == FaultType.ARBITER),
        )
        observation = DiagnosticObservation(
            query=query, original_corpus=original_corpus,
            specialist_traces=specialist_traces, final_verdict=final_verdict,
        )
        predicted_layer = self.diagnose(observation)
        success = AnswerScorer.score_answer(final_verdict, gold_answer, required_claims)

        trace = ExecutionTrace(
            query_id=query_id, query=query, gold_answer=gold_answer,
            required_claims=required_claims, condition=condition_key,
            original_corpus=original_corpus, specialist_traces=specialist_traces,
            final_verdict=final_verdict, injected_fault_layer=injected_layer,
            predicted_fault_layer=predicted_layer, is_failed_trajectory=not success,
        )
        if write_log:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = self.logs_dir / f"{query_id}_{condition_key}_{injected_layer.value}_{timestamp}.json"
            filename.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
        return trace
