# tests/test_pipeline.py
import pytest
from src.pipeline.agent_runner import BaselinePipeline
from src.schema import FaultType
from src.evaluation.scoring import AnswerScorer

def test_canonical_scoring_edge_cases():
    gold = "The Trade Oversight Commission (TOC)."
    
    # Equivalent phrasing
    assert AnswerScorer.score_answer("Authorized by the Trade Oversight Commission (TOC).", gold) is True
    assert AnswerScorer.score_answer("Authorized by the TOC.", gold) is True
    
    # Negation (should fail)
    assert AnswerScorer.score_answer("The Trade Oversight Commission did not authorize it.", gold) is False
    
    # Wrong entity (should fail)
    assert AnswerScorer.score_answer("Authorized by the Department of Treasury.", gold) is False
    
    # Abstention / Missing (should fail)
    assert AnswerScorer.score_answer("There is insufficient evidence to determine the agency.", gold) is False

def test_pipeline_repair_execution():
    pipeline = BaselinePipeline()
    
    # 1. Run baseline with an injected retrieval fault (causing failure)
    faulty_trace = pipeline.execute_pipeline("data/seed_sample.json", "A_lowC_lowP", FaultType.RETRIEVAL)
    original_correct = AnswerScorer.score_answer(faulty_trace.final_verdict, faulty_trace.gold_answer)
    assert original_correct is False
    
    # 2. Apply intervention by running without the fault (remediated baseline)
    repaired_trace = pipeline.execute_pipeline("data/seed_sample.json", "A_lowC_lowP", FaultType.NONE)
    intervention_correct = AnswerScorer.score_answer(repaired_trace.final_verdict, repaired_trace.gold_answer)
    assert intervention_correct is True
    
    # 3. Formally verify repair condition: Not Correct(base) AND Correct(intervention)
    is_repaired = (not original_correct) and intervention_correct
    assert is_repaired is True

def test_independent_fault_diagnosis():
    pipeline = BaselinePipeline()
    
    # Execute with injected specialist fault
    trace = pipeline.execute_pipeline("data/seed_sample.json", "A_lowC_lowP", FaultType.SPECIALIST)
    
    # Mock diagnostic classifier checking trace state independently of final correctness
    predicted_layer = FaultType.SPECIALIST if any("[INJECTED FAULT]" in s.raw_output for s in trace.specialist_traces) else FaultType.NONE
    
    # Verify diagnosis matches ground truth independently
    assert predicted_layer == trace.injected_fault_layer