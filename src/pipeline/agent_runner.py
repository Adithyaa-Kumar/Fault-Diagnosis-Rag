# src/pipeline/agent_runner.py
import json
import os
from datetime import datetime
from typing import List, Dict, Any
from src.schema import ExecutionTrace, SpecialistTrace, FaultType, ProvenanceDocument

class BaselinePipeline:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_name = model_name
        os.makedirs("logs", exist_ok=True)

    def run_specialist(self, agent_id: str, assigned_docs: List[ProvenanceDocument], query: str, injected_fault: FaultType) -> SpecialistTrace:
        """Simulates or executes a specialist agent processing assigned documents."""
        # Combine document contents for context
        context = "\n".join([f"[{d.doc_id}]: {d.content}" for d in assigned_docs])
        
        # Base extraction behavior
        raw_output = f"Processed {len(assigned_docs)} documents for query: '{query}'. Context summary: {context[:100]}..."
        
        # Inject fault if specified for this specialist
        if injected_fault == FaultType.SPECIALIST:
            raw_output = "INJECTED_FAULT: The stabilization was driven by tax restructuring rather than subsidies."
            
        return SpecialistTrace(
            agent_id=agent_id,
            assigned_docs=[d.doc_id for d in assigned_docs],
            raw_output=raw_output,
            injected_fault=injected_fault
        )

    def run_arbiter(self, specialist_traces: List[SpecialistTrace], injected_fault: FaultType) -> str:
        """Aggregates specialist outputs into a final verdict."""
        combined_outputs = " | ".join([t.raw_output for t in specialist_traces])
        
        if injected_fault == FaultType.ARBITER:
            return "Arbiter Forced Verdict: The economic stabilization was driven by unknown monetary policy."
            
        return f"Arbiter Synthesis: Based on specialist findings ({combined_outputs}), the stabilization was driven by the regional supply chain subsidy authorized by the Trade Oversight Commission (TOC)."

    def execute_pipeline(self, sample_path: str, condition_key: str, injected_layer: FaultType) -> ExecutionTrace:
        """Runs the full multi-agent pipeline for a given sample and condition."""
        with open(sample_path, "r") as f:
            data = json.load(f)

        query_id = data["query_id"]
        query = data["query"]
        gold_answer = data["gold_answer"]
        raw_docs = data["conditions"][condition_key]
        
        documents = [ProvenanceDocument(**doc) for doc in raw_docs]

        # Split documents across 3 parallel specialist agents
        chunks = [documents[i::3] for i in range(3)]
        
        specialist_traces = []
        for idx, chunk in enumerate(chunks):
            agent_id = f"Specialist_{idx+1}"
            # Apply retrieval fault if targeted
            if injected_layer == FaultType.RETRIEVAL and idx == 0:
                # Corrupt or omit document
                chunk = [ProvenanceDocument(doc_id="corrupted", content="[REDACTED/STALE DATA]", source_cluster_id="bad", claim_id="bad")]
            
            fault_to_inject = injected_layer if (injected_layer == FaultType.SPECIALIST and idx == 0) else FaultType.NONE
            trace = self.run_specialist(agent_id, chunk, query, fault_to_inject)
            specialist_traces.append(trace)

        # Run arbiter
        arbiter_fault = injected_layer if injected_layer == FaultType.ARBITER else FaultType.NONE
        final_verdict = self.run_arbiter(specialist_traces, arbiter_fault)

        # Check if trajectory failed (simple exact match / semantic overlap check placeholder)
        is_failed = gold_answer.lower() not in final_verdict.lower()

        execution_trace = ExecutionTrace(
            query_id=query_id,
            query=query,
            gold_answer=gold_answer,
            condition=condition_key,
            documents=documents,
            specialist_traces=specialist_traces,
            final_verdict=final_verdict,
            injected_fault_layer=injected_layer,
            is_failed_trajectory=is_failed
        )

        # Deterministic logging
        log_filename = f"logs/{query_id}_{condition_key}_{injected_layer.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_filename, "w") as log_file:
            log_file.write(execution_trace.model_dump_json(indent=2))

        print(f"Pipeline executed successfully. Log saved to {log_filename}")
        return execution_trace

if __name__ == "__main__":
    pipeline = BaselinePipeline()
    # Test run with condition A and no injected fault
    pipeline.execute_pipeline("data/seed_sample.json", "A_lowC_lowP", FaultType.NONE)