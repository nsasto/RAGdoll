# Test all hybrid modes quickly with a small subset
# Usage: .\test_hybrid_modes.ps1

param(
    [int]$Subset = 25  # Small subset for quick testing
)

$ErrorActionPreference = "Stop"

# Load .env file from project root
$envFile = Join-Path (Split-Path $PSScriptRoot -Parent) ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)\s*=\s*(.+)\s*$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            $value = $value -replace '^["'']|["'']$'
            [Environment]::SetEnvironmentVariable($name, $value, 'Process')
        }
    }
    Write-Host "[OK] Loaded .env" -ForegroundColor Green
}

$dataset = "2wikimultihopqa"
$modes = @("concat", "expand", "weighted", "rerank")
$separator = "=" * 70
Write-Host ""
Write-Host $separator -ForegroundColor Cyan
Write-Host "Hybrid Mode Comparison Test" -ForegroundColor Cyan
Write-Host $separator -ForegroundColor Cyan
Write-Host ""
Write-Host "Dataset: $dataset" -ForegroundColor Green
Write-Host "Subset:  $Subset queries" -ForegroundColor Green
Write-Host "Modes:   $($modes -join ', ')" -ForegroundColor Green
Write-Host ""

# Create index once (reuse for all modes)
# Note: benchmark creates shared graph index at db\ragdoll_${dataset}_${Subset}_graph\
$indexPath = "db\ragdoll_${dataset}_${Subset}_graph"
if (-not (Test-Path "$indexPath\graph.pkl")) {
    Write-Host "Creating hybrid index (one-time setup)..." -ForegroundColor Yellow
    python ragdoll_benchmark.py -d $dataset -n $Subset --mode hybrid --create
    Write-Host ""
}
else {
    Write-Host "Using existing index at $indexPath" -ForegroundColor Green
    Write-Host ""
}

# Test each mode
$results = @()

foreach ($mode in $modes) {
    Write-Host $separator -ForegroundColor Cyan
    Write-Host "Testing mode: $mode" -ForegroundColor Cyan
    Write-Host $separator -ForegroundColor Cyan
    
    $env:HYBRID_MODE = $mode
    python ragdoll_benchmark.py -d $dataset -n $Subset --mode hybrid --benchmark
    
    # Read results immediately (before next mode overwrites the file)
    $resultFile = "results\ragdoll_${dataset}_${Subset}_hybrid.json"
    if (Test-Path $resultFile) {
        $result = Get-Content $resultFile | ConvertFrom-Json
        
        # Verify which mode was actually used
        $actualMode = if ($result.config.hybrid_mode) { $result.config.hybrid_mode } else { "unknown" }
        if ($actualMode -ne $mode) {
            Write-Host "WARNING: Expected mode '$mode' but result shows '$actualMode'" -ForegroundColor Yellow
        }
        
        $results += [PSCustomObject]@{
            Mode             = $mode
            PerfectRetrieval = "{0:P1}" -f $result.metrics.perfect_retrieval_rate
            Recall8          = if ($result.metrics.recall_at_k) { "{0:P1}" -f $result.metrics.recall_at_k } else { "N/A" }
            MRR              = "{0:F3}" -f $result.metrics.mrr
            LatencyMean      = if ($result.metrics.latency.mean_ms) { "{0:F0}ms" -f [double]$result.metrics.latency.mean_ms } else { "N/A" }
        }
        
        # Copy result to mode-specific file for reference
        $modeSpecificFile = "results\ragdoll_${dataset}_${Subset}_hybrid_${mode}.json"
        Copy-Item $resultFile $modeSpecificFile -Force
    }
    else {
        Write-Host "WARNING: Result file not found: $resultFile" -ForegroundColor Red
    }
    
    Write-Host ""
}

# Display comparison table
Write-Host ""
Write-Host $separator -ForegroundColor Green
Write-Host "HYBRID MODE COMPARISON RESULTS" -ForegroundColor Green
Write-Host $separator -ForegroundColor Green
Write-Host ""

$results | Format-Table -AutoSize

Write-Host ""
Write-Host "Recommendation: Use the mode with highest Perfect Retrieval Rate" -ForegroundColor Cyan
Write-Host ""

# Find best mode
$best = $results | Sort-Object { [double]($_.PerfectRetrieval -replace '%', '') } -Descending | Select-Object -First 1
Write-Host "Best mode: $($best.Mode) with $($best.PerfectRetrieval) perfect retrieval" -ForegroundColor Green
Write-Host ""
