from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class FaultType(str, Enum):
    NONE = "none"
    RETRIEVAL = "retrieval"
    SPECIALIST = "specialist"
    ARBITER = "arbiter"


class ProvenanceDocument(BaseModel):
    doc_id: str
    content: str
    source_cluster_id: str
    claim_id: str


class SpecialistTrace(BaseModel):
    agent_id: str
    assigned_docs: List[str]
    retrieved_docs: List[ProvenanceDocument]
    actual_inputs: List[str]
    raw_output: str


class DiagnosticObservation(BaseModel):
    """Only observable runtime evidence; deliberately excludes injected labels."""
    query: str
    original_corpus: List[ProvenanceDocument]
    specialist_traces: List[SpecialistTrace]
    final_verdict: str


class ExecutionTrace(BaseModel):
    query_id: str
    query: str
    gold_answer: str
    required_claims: List[Dict[str, Any]] = Field(default_factory=list)
    condition: str
    original_corpus: List[ProvenanceDocument]
    specialist_traces: List[SpecialistTrace]
    final_verdict: str
    injected_fault_layer: FaultType  # Evaluator-only ground truth; never passed to diagnosis.
    predicted_fault_layer: Optional[FaultType] = None
    is_failed_trajectory: bool = False
    is_repaired: bool = False
