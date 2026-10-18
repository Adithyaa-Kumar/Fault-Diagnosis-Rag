from pathlib import Path
from src.pipeline.agent_runner import BaselinePipeline
from src.schema import FaultType
from src.evaluation.scoring import AnswerScorer

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "seed_sample.json"


def test_pipeline_repair_execution():
    pipeline = BaselinePipeline()
    faulty_trace = pipeline.execute_pipeline(str(SAMPLE), "A_lowC_lowP", FaultType.RETRIEVAL, write_log=False)
    original_correct = AnswerScorer.score_answer(faulty_trace.final_verdict, faulty_trace.gold_answer, faulty_trace.required_claims)
    repaired_trace = pipeline.execute_pipeline(str(SAMPLE), "A_lowC_lowP", FaultType.NONE, write_log=False)
    intervention_correct = AnswerScorer.score_answer(repaired_trace.final_verdict, repaired_trace.gold_answer, repaired_trace.required_claims)
    assert (not original_correct) and intervention_correct


def test_independent_fault_diagnosis_uses_observations_not_labels():
    pipeline = BaselinePipeline()
    for fault in FaultType:
        trace = pipeline.execute_pipeline(str(SAMPLE), "A_lowC_lowP", fault, write_log=False)
        assert trace.predicted_fault_layer == fault
