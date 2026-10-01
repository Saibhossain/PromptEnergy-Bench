<#
.SYNOPSIS
    Pipeline Verification & Smoke Test Script for PromptEnergy-Bench.

.DESCRIPTION
    Executes a complete, end-to-end verification run across:
    - All 4 Benchmark Datasets: gsm8k, natural_questions, contexteval, cnn_dailymail
    - All 3 Benchmark Experiments: Prompting Strategies (Exp 1), Context Scaling (Exp 2), BM25 RAG (Exp 3)
    - 3 Target Models: qwen3.5:2b, qwen3.5:0.8b, gemma3:4b
    - Evaluation Sample Size: 10 samples per condition
    - Hardware: windows_10core_pc (or windows_cuda_pc if GPU is available)

.EXAMPLE
    .\scripts\test_pipeline_run.ps1
    .\scripts\test_pipeline_run.ps1 -Hardware "windows_cuda_pc"
    .\scripts\test_pipeline_run.ps1 -EvalSize 5
#>

param(
    [string]$Hardware = "windows_10core_pc",
    [string]$Models = "qwen3.5:2b,qwen3.5:0.8b,gemma3:4b",
    [string]$Dataset = "all",
    [string]$EvalSize = "10",
    [int]$RunsPerCondition = 1,
    [string]$Experiment = "all",
    [switch]$ForceRerun = $false
)

Write-Host "===========================================================================" -ForegroundColor Cyan
Write-Host " PROMPTENERGY-BENCH: PIPELINE VALIDATION & SMOKE TEST RUNNER" -ForegroundColor Cyan
Write-Host "===========================================================================" -ForegroundColor Cyan
Write-Host " Hardware Preset     : $Hardware"
Write-Host " Target Models       : $Models"
Write-Host " Target Datasets     : $Dataset"
Write-Host " Evaluation Size     : $EvalSize samples per dataset"
Write-Host " Repetitions/Cond    : $RunsPerCondition"
Write-Host " Target Experiments  : $Experiment"
Write-Host " Force Rerun         : $ForceRerun"
Write-Host "===========================================================================" -ForegroundColor Cyan

# 1. Determine Python Executable
$PythonExe = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
    $PythonExe = ".\.venv\Scripts\python.exe"
}

Write-Host "`n[1/3] Using Python: $PythonExe" -ForegroundColor Green

# 2. Check Ollama Service
Write-Host "`n[2/3] Checking Ollama Service and Models..." -ForegroundColor Green
try {
    $ollamaList = ollama list
    Write-Host "Active Ollama Models:"
    Write-Host $ollamaList
} catch {
    Write-Warning "Could not query 'ollama list'. Please make sure the Ollama service is running."
}

# 3. Build Command Arguments
$cmdArgs = @(
    "scripts/run_all_experiments.py",
    "--hardware", $Hardware,
    "--models", $Models,
    "--dataset", $Dataset,
    "--experiment", $Experiment,
    "--eval-size", $EvalSize,
    "--runs-per-condition", $RunsPerCondition
)

if (-not $ForceRerun) {
    $cmdArgs += "--skip-existing"
}

Write-Host "`n[3/3] Executing Master Experiment Pipeline..." -ForegroundColor Green
Write-Host "Command: $PythonExe $($cmdArgs -join ' ')" -ForegroundColor Gray
Write-Host "===========================================================================`n" -ForegroundColor Cyan

& $PythonExe $cmdArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n===========================================================================" -ForegroundColor Green
    Write-Host " [SUCCESS] All pipeline tests completed successfully!" -ForegroundColor Green
    Write-Host " Deliverables generated under: results/" -ForegroundColor Green
    Write-Host " Master comparison report    : results/master_comparison/" -ForegroundColor Green
    Write-Host "===========================================================================" -ForegroundColor Green
} else {
    Write-Host "`n===========================================================================" -ForegroundColor Red
    Write-Host " [ERROR] Pipeline test exited with code $LASTEXITCODE. Check error messages above." -ForegroundColor Red
    Write-Host "===========================================================================" -ForegroundColor Red
}
