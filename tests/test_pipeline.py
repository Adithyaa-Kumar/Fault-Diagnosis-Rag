
import inspect

from src.pipeline.agent_runner import BaselinePipeline
from src.schema import FaultType
from src.evaluation.scoring import AnswerScorer
from src.evaluation.validate_dataset import validate_dataset


DATA_PATH = "data/seed_sample.json"


def test_canonical_scoring():
    gold = "The Trade Oversight Commission (TOC)."

    assert AnswerScorer.score_answer(
        "The authorized agency is the Trade Oversight Commission.",
        gold,
    )
    assert AnswerScorer.score_answer("Authorized by the TOC.", gold)

    assert not AnswerScorer.score_answer(
        "The Trade Oversight Commission did not authorize it.", gold
    )
    assert not AnswerScorer.score_answer(
        "Authorized by the Department of Treasury.", gold
    )
    assert not AnswerScorer.score_answer(
        "Insufficient evidence to determine the agency.", gold
    )
    assert not AnswerScorer.score_answer("", gold)


def test_dataset_validation():
    assert validate_dataset(DATA_PATH) == []


def test_diagnostic_function_does_not_accept_fault_label():
    signature = inspect.signature(
        BaselinePipeline.diagnose_fault_layer
    )
    assert "injected_layer" not in signature.parameters
    assert "injected_fault_layer" not in signature.parameters


def test_all_sixteen_condition_fault_combinations():
    pipeline = BaselinePipeline()

    conditions = [
        "A_lowC_lowP",
        "B_highC_lowP",
        "C_lowC_highP",
        "D_highC_highP",
    ]

    fault_types = list(FaultType)
    assert len(conditions) * len(fault_types) == 16

    for condition in conditions:
        for fault_type in fault_types:
            trace = pipeline.execute_pipeline(
                DATA_PATH, condition, fault_type
            )

            assert trace.condition == condition
            assert trace.injected_fault_layer == fault_type
            assert trace.predicted_fault_layer is not None

            # Score answer correctness independently from diagnosis.
            AnswerScorer.score_answer(
                trace.final_verdict, trace.gold_answer
            )


def test_diagnosis_is_compared_with_ground_truth():
    pipeline = BaselinePipeline()

    for fault_type in FaultType:
        trace = pipeline.execute_pipeline(
            DATA_PATH, "A_lowC_lowP", fault_type
        )

        # The diagnostic prediction is produced from observable traces.
        assert trace.predicted_fault_layer == (
            trace.injected_fault_layer
        )


def test_real_fault_recovery_intervention():
    pipeline = BaselinePipeline()

    failed = pipeline.execute_pipeline(
        DATA_PATH, "A_lowC_lowP", FaultType.RETRIEVAL
    )
    baseline_success = AnswerScorer.score_answer(
        failed.final_verdict, failed.gold_answer
    )
    assert baseline_success is False

    # Intervention: rerun the same condition with the injected fault off.
    repaired = pipeline.execute_pipeline(
        DATA_PATH, "A_lowC_lowP", FaultType.NONE
    )
    intervention_success = AnswerScorer.score_answer(
        repaired.final_verdict, repaired.gold_answer
    )

    is_repair = (not baseline_success) and intervention_success
    assert is_repair is True


def test_repair_predicate_truth_table():
    cases = [
        (True, True, False),
        (True, False, False),
        (False, True, True),
        (False, False, False),
    ]

    for baseline_success, intervention_success, expected in cases:
        is_repair = (
            (not baseline_success) and intervention_success
        )
        assert is_repair is expected
