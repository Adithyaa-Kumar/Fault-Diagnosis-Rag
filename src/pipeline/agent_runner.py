
import json
import os
from datetime import datetime
from typing import List, Optional

from src.schema import (
    ExecutionTrace,
    SpecialistTrace,
    FaultType,
    ProvenanceDocument,
)


class BaselinePipeline:
    def __init__(self, model_name: str = "mock-pipeline"):
        self.model_name = model_name
        os.makedirs("logs", exist_ok=True)

    def run_specialist(
        self,
        agent_id: str,
        assigned_docs: List[ProvenanceDocument],
        retrieved_docs: List[ProvenanceDocument],
        query: str,
        inject_specialist_fault: bool = False,
    ) -> SpecialistTrace:
        context = "\n".join(
            f"[{doc.doc_id}]: {doc.content}"
            for doc in assigned_docs
        )

        actual_inputs = [
            f"Query: {query}",
            f"Context: {context}",
        ]

        if inject_specialist_fault:
            raw_output = (
                "Specialist conclusion: "
                "The Department of Treasury authorized the stabilization."
            )
        else:
            raw_output = f"Specialist extracted context: {context}"

        return SpecialistTrace(
            agent_id=agent_id,
            assigned_docs=[doc.doc_id for doc in assigned_docs],
            retrieved_docs=retrieved_docs,
            actual_inputs=actual_inputs,
            raw_output=raw_output,
        )

    def run_arbiter(
        self,
        specialist_traces: List[SpecialistTrace],
        inject_arbiter_fault: bool = False,
    ) -> str:
        combined_text = " ".join(
            trace.raw_output for trace in specialist_traces
        ).lower()

        if inject_arbiter_fault:
            return (
                "Arbiter Verdict: "
                "The Department of Treasury authorized the action."
            )

        if "trade oversight commission" in combined_text or (
            " toc " in f" {combined_text} "
        ):
            return (
                "Arbiter Verdict: Based on the specialist evidence, "
                "the authorized agency is the Trade Oversight Commission."
            )

        return (
            "Arbiter Verdict: Insufficient evidence to determine "
            "the authorizing agency."
        )

    @staticmethod
    def diagnose_fault_layer(
        specialist_traces: List[SpecialistTrace],
        final_verdict: str,
        original_corpus: List[ProvenanceDocument],
    ) -> FaultType:
        """
        Diagnostic mock.

        IMPORTANT:
        This function receives observations only. It does not receive
        the injected fault label or an ExecutionTrace containing it.
        """
        corpus_ids = {doc.doc_id for doc in original_corpus}

        # Retrieval anomaly: an assigned document is absent from the
        # original corpus. This uses observed trace state, not the label.
        for trace in specialist_traces:
            if any(doc_id not in corpus_ids for doc_id in trace.assigned_docs):
                return FaultType.RETRIEVAL

        verdict_lower = final_verdict.lower()

        # Specialist anomaly: an individual specialist asserts the
        # wrong agency.
        for trace in specialist_traces:
            output = trace.raw_output.lower()
            if (
                "department of treasury" in output
                or "ministry of finance" in output
            ):
                return FaultType.SPECIALIST

        # Arbiter anomaly: specialist evidence supports the TOC, but
        # the final verdict conflicts with it.
        specialists_support_toc = any(
            "trade oversight commission" in trace.raw_output.lower()
            or " toc " in f" {trace.raw_output.lower()} "
            for trace in specialist_traces
        )
        arbiter_says_wrong_agency = (
            "department of treasury" in verdict_lower
            or "ministry of finance" in verdict_lower
        )

        if specialists_support_toc and arbiter_says_wrong_agency:
            return FaultType.ARBITER

        return FaultType.NONE

    def execute_pipeline(
        self,
        sample_path: str,
        condition_key: str,
        injected_layer: FaultType,
        documents_override: Optional[list] = None,
    ) -> ExecutionTrace:
        with open(sample_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        query_id = data["query_id"]
        query = data["query"]
        gold_answer = data["gold_answer"]

        raw_docs = (
            documents_override
            if documents_override is not None
            else data["conditions"][condition_key]
        )

        documents = [
            doc if isinstance(doc, ProvenanceDocument)
            else ProvenanceDocument(**doc)
            for doc in raw_docs
        ]

        # Keep three specialist slots. Empty input is allowed when a
        # document has been removed by an intervention.
        chunks = [
            [documents[i]] if i < len(documents) else []
            for i in range(3)
        ]

        specialist_traces = []

        for idx, original_chunk in enumerate(chunks):
            assigned_chunk = list(original_chunk)

            # Inject retrieval corruption without exposing a fault label
            # to the diagnostic function.
            if injected_layer == FaultType.RETRIEVAL:
                assigned_chunk = [
                    ProvenanceDocument(
                        doc_id=f"corrupted_retrieval_{idx}",
                        content="Irrelevant background information unrelated "
                                "to the requested authorizing agency.",
                        source_cluster_id="corrupted_source",
                        claim_id="irrelevant_claim",
                    )
                ]

            trace = self.run_specialist(
                agent_id=f"Specialist_{idx + 1}",
                assigned_docs=assigned_chunk,
                retrieved_docs=list(assigned_chunk),
                query=query,
                inject_specialist_fault=(
                    injected_layer == FaultType.SPECIALIST and idx == 0
                ),
            )
            specialist_traces.append(trace)

        final_verdict = self.run_arbiter(
            specialist_traces,
            inject_arbiter_fault=(injected_layer == FaultType.ARBITER),
        )

        # Diagnose BEFORE attaching ground-truth metadata to the trace.
        predicted_layer = self.diagnose_fault_layer(
            specialist_traces=specialist_traces,
            final_verdict=final_verdict,
            original_corpus=documents,
        )

        # Ground truth is retained only for evaluator-side comparison.
        from src.evaluation.scoring import AnswerScorer

        is_failed = not AnswerScorer.score_answer(
            final_verdict, gold_answer
        )

        execution_trace = ExecutionTrace(
            query_id=query_id,
            query=query,
            gold_answer=gold_answer,
            condition=condition_key,
            original_corpus=documents,
            specialist_traces=specialist_traces,
            final_verdict=final_verdict,
            injected_fault_layer=injected_layer,
            predicted_fault_layer=predicted_layer,
            is_failed_trajectory=is_failed,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_filename = os.path.join(
            "logs",
            f"{query_id}_{condition_key}_{injected_layer.value}_{timestamp}.json",
        )

        with open(log_filename, "w", encoding="utf-8") as log_file:
            log_file.write(execution_trace.model_dump_json(indent=2))

        return execution_trace
