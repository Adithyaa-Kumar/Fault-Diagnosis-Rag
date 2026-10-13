# src/schema.py
from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class FaultType(str, Enum):
    NONE = "none"
    RETRIEVAL = "retrieval"
    SPECIALIST = "specialist"
    ARBITER = "arbiter"

class ProvenanceDocument(BaseModel):
    doc_id: str
    content: str
    source_cluster_id: str  # Tracks underlying provenance dependence (P)
    claim_id: str           # Tracks content redundancy (C)

class SpecialistTrace(BaseModel):
    agent_id: str
    assigned_docs: List[str]
    raw_output: str
    injected_fault: FaultType = FaultType.NONE

class ExecutionTrace(BaseModel):
    query_id: str
    query: str
    gold_answer: str
    condition: str          # A (Low C/Low P), B (High C/Low P), C (Low C/High P), D (High C/High P)
    documents: List[ProvenanceDocument]
    specialist_traces: List[SpecialistTrace]
    final_verdict: str
    gold_log_prob: Optional[float] = None
    injected_fault_layer: FaultType
    is_failed_trajectory: bool = False