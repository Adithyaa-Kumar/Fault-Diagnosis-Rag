# tests/test_pipeline.py
import pytest
from src.pipeline.agent_runner import BaselinePipeline
from src.schema import FaultType

def test_baseline_success():
    pipeline = BaselinePipeline()
    trace = pipeline.execute_pipeline("data/seed_sample.json", "A_lowC_lowP", FaultType.NONE)
    assert trace.is_failed_trajectory is False
    assert "Trade Oversight Commission" in trace.final_verdict

def test_retrieval_fault_failure():
    pipeline = BaselinePipeline()
    trace = pipeline.execute_pipeline("data/seed_sample.json", "A_lowC_lowP", FaultType.RETRIEVAL)
    assert trace.is_failed_trajectory is True

def test_arbiter_fault_failure():
    pipeline = BaselinePipeline()
    trace = pipeline.execute_pipeline("data/seed_sample.json", "A_lowC_lowP", FaultType.ARBITER)
    assert trace.is_failed_trajectory is True