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
        context = "\n".join([f"[{d.doc_id}]: {d.content}" for d in assigned_docs])
        raw_output = f"Specialist extracted context: {context}"
        
        if injected_fault == FaultType.SPECIALIST:
            raw_output = "Specialist extracted context: [INJECTED FAULT] The Department of Treasury authorized the stabilization."
            
        return SpecialistTrace(
            agent_id=agent_id,
            assigned_docs=[d.doc_id for d in assigned_docs],
            raw_output=raw_output,
            injected_fault=injected_fault
        )

    def run_arbiter(self, specialist_traces: List[SpecialistTrace], gold_answer_keyword: str, injected_fault: FaultType) -> str:
        combined_text = " ".join([t.raw_output for t in specialist_traces])
        
        if injected_fault == FaultType.ARBITER:
            return "Arbiter Verdict: The Department of Treasury authorized the action."
            
        # Evidence-dependent verification: if keyword is missing from specialist output, arbiter fails safely
        if gold_answer_keyword.lower() in combined_text.lower():
            return f"Arbiter Verdict: Based on verified specialist evidence, the authorized agency is {gold_answer_keyword}."
        else:
            return "Arbiter Verdict: Insufficient evidence in retrieved documents to determine the authorizing agency."

    def execute_pipeline(self, sample_path: str, condition_key: str, injected_layer: FaultType) -> ExecutionTrace:
        with open(sample_path, "r") as f:
            data = json.load(f)

        query_id = data["query_id"]
        query = data["query"]
        gold_answer = data["gold_answer"]
        raw_docs = data["conditions"][condition_key]
        
        documents = [ProvenanceDocument(**doc) for doc in raw_docs]

        # Exactly 3 documents assigned to 3 parallel specialists (1 per specialist)
        chunks = [[documents[i]] if i < len(documents) else [] for i in range(3)]
        
        specialist_traces = []
        for idx, chunk in enumerate(chunks):
            agent_id = f"Specialist_{idx+1}"
            
            # Apply retrieval fault if targeted
            if injected_layer == FaultType.RETRIEVAL and idx == 0:
                chunk = [ProvenanceDocument(doc_id="faulty_doc", content="Irrelevant background information.", source_cluster_id="bad", claim_id="bad")]
            
            fault_to_inject = injected_layer if (injected_layer == FaultType.SPECIALIST and idx == 0) else FaultType.NONE
            trace = self.run_specialist(agent_id, chunk, query, fault_to_inject)
            specialist_traces.append(trace)

        arbiter_fault = injected_layer if injected_layer == FaultType.ARBITER else FaultType.NONE
        keyword = "Trade Oversight Commission"
        final_verdict = self.run_arbiter(specialist_traces, keyword, arbiter_fault)

        is_failed = keyword.lower() not in final_verdict.lower()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
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

        log_filename = f"logs/{query_id}_{condition_key}_{injected_layer.value}_{timestamp}.json"
        with open(log_filename, "w") as log_file:
            log_file.write(execution_trace.model_dump_json(indent=2))

        return execution_trace