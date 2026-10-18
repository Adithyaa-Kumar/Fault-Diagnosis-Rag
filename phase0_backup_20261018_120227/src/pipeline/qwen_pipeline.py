"""Real local-Qwen version of the controlled parallel fan-in pipeline.

Fault labels are used only by the intervention injector. The diagnostic method
receives DiagnosticObservation, which deliberately has no gold answer or label.
The shared model is called sequentially to fit the 6 GB GPU; the architecture
is parallel fan-in (three isolated specialist contexts followed by an arbiter).
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union

from src.evaluation.scoring import AnswerScorer
from src.pipeline.agent_runner import BaselinePipeline
from src.schema import DiagnosticObservation, ExecutionTrace, FaultType, ProvenanceDocument, SpecialistTrace


class QwenParallelPipeline:
    def __init__(self, model=None, logs_dir: str = "logs"):
        if model is None:
            from src.models.local_qwen import LocalQwen
            model = LocalQwen()
        self.model = model
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _generate(self, context: str, question: str, system_prompt: str, max_new_tokens: int = 96) -> dict:
        return self.model.generate_answer(context=context, question=question,
                                          system_prompt=system_prompt,
                                          max_new_tokens=max_new_tokens)

    def execute_pipeline(self, sample_path: str, condition_key: str, injected_layer: FaultType,
                         documents_override: Optional[Sequence[Union[ProvenanceDocument, dict]]] = None,
                         write_log: bool = True) -> ExecutionTrace:
        data = json.loads(Path(sample_path).read_text(encoding="utf-8"))
        docs_raw = documents_override if documents_override is not None else data["conditions"][condition_key]
        docs = [d if isinstance(d, ProvenanceDocument) else ProvenanceDocument(**d) for d in docs_raw]
        original_docs = [ProvenanceDocument(**d) for d in data["conditions"][condition_key]]
        specialist_traces = []
        for idx in range(3):
            original_chunk = [
                doc for position, doc in enumerate(docs)
                if position % 3 == idx
            ]
            actual_chunk = list(original_chunk)
            if injected_layer == FaultType.RETRIEVAL and idx == 0:
                actual_chunk = [ProvenanceDocument(doc_id="faulty_doc",
                    content="Irrelevant background information unrelated to the question.",
                    source_cluster_id="injected_irrelevant_source", claim_id="claim_irrelevant")]
            context = "\n".join(f"[{d.doc_id}] {d.content}" for d in actual_chunk) or "No evidence provided."
            generated = self._generate(context, data["query"],
                "You are a specialist in a parallel evidence-reasoning system. Use only the supplied context. "
                "Extract the relevant facts that help answer the question. If evidence is missing, say so. "
                "Do not guess. Be concise.", 96)
            output = generated["answer"]
            if injected_layer == FaultType.SPECIALIST and idx == 0:
                output = "The Department of Treasury authorized the action."
            specialist_traces.append(SpecialistTrace(
                agent_id=f"Specialist_{idx+1}",
                assigned_docs=[d.doc_id for d in actual_chunk],
                retrieved_docs=list(actual_chunk),
                actual_inputs=[f"Query: {data['query']}", f"Context: {context}"],
                raw_output=output,
            ))

        specialist_context = "\n\n".join(
            f"{t.agent_id} report:\n{t.raw_output}" for t in specialist_traces)
        final_verdict = self._generate(specialist_context, data["query"],
            "You are the arbiter in a parallel retrieval-augmented system. Answer the user's question "
            "using only the specialist reports. Combine supported facts, do not invent missing facts, "
            "and state uncertainty when evidence is insufficient. Return a direct answer.", 128)["answer"]
        if injected_layer == FaultType.ARBITER:
            final_verdict = "The Department of Treasury authorized the action."

        observation = DiagnosticObservation(query=data["query"], original_corpus=original_docs,
            specialist_traces=specialist_traces, final_verdict=final_verdict)
        predicted = BaselinePipeline.diagnose(observation)
        claims = data.get("required_claims", [])
        score = AnswerScorer.score_metrics(final_verdict, data["gold_answer"], claims or None)
        trace = ExecutionTrace(query_id=data["query_id"], query=data["query"],
            gold_answer=data["gold_answer"], required_claims=claims, condition=condition_key,
            original_corpus=original_docs, specialist_traces=specialist_traces,
            final_verdict=final_verdict, injected_fault_layer=injected_layer,
            predicted_fault_layer=predicted, is_failed_trajectory=not score["correct"])
        if write_log:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            (self.logs_dir / f"{trace.query_id}_{condition_key}_{injected_layer.value}_qwen_{stamp}.json").write_text(
                trace.model_dump_json(indent=2), encoding="utf-8")
        return trace
