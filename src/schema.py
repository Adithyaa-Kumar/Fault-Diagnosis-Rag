
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


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


class ExecutionTrace(BaseModel):
    query_id: str
    query: str
    gold_answer: str
    condition: str
    original_corpus: List[ProvenanceDocument]
    specialist_traces: List[SpecialistTrace]
    final_verdict: str

    # These two fields are for evaluator/reporting purposes.
    # Never pass this full object to diagnose_fault_layer().
    injected_fault_layer: FaultType
    predicted_fault_layer: Optional[FaultType] = None

    is_failed_trajectory: bool = False
    is_repaired: bool = False
