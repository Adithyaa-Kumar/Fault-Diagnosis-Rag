$ErrorActionPreference = "Stop"
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
python -m src.evaluation.validate_dataset
if ($LASTEXITCODE -ne 0) { throw "Dataset validation failed" }
python -m src.evaluation.run_qwen_factorial
if ($LASTEXITCODE -ne 0) { throw "Real-Qwen factorial failed" }
Write-Host "Phase 0 engineering gate passed. Proceed to Phase 1 with MultiHop-RAG."
