# RAGdoll Benchmark Runner
# Automates benchmark execution and comparison

param(
    [int]$Subset = 51,
    [string]$Dataset = "2wikimultihopqa",
    [switch]$CreateOnly,
    [switch]$BenchmarkOnly,
    [switch]$SkipBaseline
)

# Load .env file from project root if it exists
$envFile = Join-Path (Split-Path $PSScriptRoot -Parent) ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)\s*=\s*(.+)\s*$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            # Remove quotes if present
            $value = $value -replace '^["'']|["'']$'
            [Environment]::SetEnvironmentVariable($name, $value, 'Process')
        }
    }
    Write-Host "[OK] Loaded .env from project root" -ForegroundColor Green
}

# Check for OPENAI_API_KEY
if (-not $env:OPENAI_API_KEY) {
    Write-Host "ERROR: OPENAI_API_KEY not set!" -ForegroundColor Red
    Write-Host "   Please set it in .env or environment variables" -ForegroundColor Yellow
    exit 1
}

$separator = "=" * 70

$vectorIndexPath = Join-Path "db" ("ragdoll_{0}_{1}_vector" -f $Dataset, $Subset)
$graphIndexPath = Join-Path "db" ("ragdoll_{0}_{1}_graph" -f $Dataset, $Subset)
$graphIndexMarker = Join-Path $graphIndexPath "graph.pkl"
$vectorCollectionPath = Join-Path $vectorIndexPath "vector"

Write-Host ""
Write-Host $separator -ForegroundColor Cyan
Write-Host "RAGdoll Benchmark Suite" -ForegroundColor Cyan
Write-Host $separator -ForegroundColor Cyan
Write-Host ""
Write-Host "Dataset: $Dataset" -ForegroundColor Green
Write-Host "Subset:  $Subset queries" -ForegroundColor Green
Write-Host ""

# Check if dataset exists
$datasetFile = Join-Path $PSScriptRoot "datasets\$Dataset.json"
if (-not (Test-Path $datasetFile)) {
    Write-Host "WARNING: Dataset not found: $datasetFile" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please download datasets from:" -ForegroundColor Yellow
    Write-Host "https://github.com/circlemind-ai/fast-graphrag/tree/main/benchmarks/datasets" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

# Estimate costs
Write-Host "Cost Estimation" -ForegroundColor Cyan
Write-Host "   This will make API calls to OpenAI" -ForegroundColor Yellow
Write-Host "   Estimated cost: ~`$0.10 - `$0.50 depending on options" -ForegroundColor Yellow
Write-Host ""

# Confirm before proceeding
$confirm = Read-Host "Continue? (y/n)"
if ($confirm -ne "y") {
    Write-Host "Cancelled" -ForegroundColor Yellow
    exit 0
}

Write-Host ""

# Run Vector Baseline
if (-not $SkipBaseline) {
    Write-Host $separator -ForegroundColor Cyan
    Write-Host "1. Vector Baseline (Chroma + OpenAI)" -ForegroundColor Cyan
    Write-Host $separator -ForegroundColor Cyan
    Write-Host ""
    
    if ($CreateOnly -or -not $BenchmarkOnly) {
        Write-Host "Creating vector index..." -ForegroundColor Green
        python vector_baseline.py -d $Dataset -n $Subset --create
    }
    
    if ($BenchmarkOnly -or -not $CreateOnly) {
        Write-Host "Running vector benchmark..." -ForegroundColor Green
        python vector_baseline.py -d $Dataset -n $Subset --benchmark
    }
    
    Write-Host ""
}

# Run RAGdoll Vector Mode
Write-Host $separator -ForegroundColor Cyan
Write-Host "2. RAGdoll (Vector Only)" -ForegroundColor Cyan
Write-Host $separator -ForegroundColor Cyan
Write-Host ""

if ($CreateOnly -or -not $BenchmarkOnly) {
    if (Test-Path $vectorCollectionPath) {
        Write-Host "Reusing existing vector index at $vectorIndexPath" -ForegroundColor Yellow
    }
    else {
        Write-Host "Creating vector index..." -ForegroundColor Green
        python ragdoll_benchmark.py -d $Dataset -n $Subset --mode vector --create
    }
}

if ($BenchmarkOnly -or -not $CreateOnly) {
    Write-Host "Running benchmark..." -ForegroundColor Green
    python ragdoll_benchmark.py -d $Dataset -n $Subset --mode vector --benchmark
}

Write-Host ""

# Run RAGdoll Hybrid Mode
Write-Host $separator -ForegroundColor Cyan
Write-Host "3. RAGdoll (PageRank Graph)" -ForegroundColor Cyan
Write-Host $separator -ForegroundColor Cyan
Write-Host ""

if ($CreateOnly -or -not $BenchmarkOnly) {
    if (Test-Path $graphIndexMarker) {
        Write-Host "Reusing existing graph index at $graphIndexPath" -ForegroundColor Yellow
    }
    else {
        Write-Host "Creating PageRank index (includes entity extraction)..." -ForegroundColor Green
        python ragdoll_benchmark.py -d $Dataset -n $Subset --mode pagerank --create
    }
}

if ($BenchmarkOnly -or -not $CreateOnly) {
    Write-Host "Running benchmark..." -ForegroundColor Green
    python ragdoll_benchmark.py -d $Dataset -n $Subset --mode pagerank --benchmark
}

Write-Host ""

Write-Host $separator -ForegroundColor Cyan
Write-Host "4. RAGdoll (Hybrid: Vector + Graph)" -ForegroundColor Cyan
Write-Host $separator -ForegroundColor Cyan
Write-Host ""

if ($CreateOnly -or -not $BenchmarkOnly) {
    if (Test-Path $graphIndexMarker) {
        Write-Host "Hybrid mode will reuse shared graph index at $graphIndexPath" -ForegroundColor Yellow
    }
    else {
        Write-Host "Creating hybrid index (this may take a while)..." -ForegroundColor Green
        python ragdoll_benchmark.py -d $Dataset -n $Subset --mode hybrid --create
    }
}

if ($BenchmarkOnly -or -not $CreateOnly) {
    Write-Host "Running benchmark..." -ForegroundColor Green
    python ragdoll_benchmark.py -d $Dataset -n $Subset --mode hybrid --benchmark
}

Write-Host ""

Write-Host ""

# Generate comparison report
if ($BenchmarkOnly -or -not $CreateOnly) {
    Write-Host $separator -ForegroundColor Cyan
    Write-Host "5. Generating Comparison Report" -ForegroundColor Cyan
    Write-Host $separator -ForegroundColor Cyan
    Write-Host ""
    
    python compare.py -d $Dataset -n $Subset
    
    Write-Host ""
    Write-Host $separator -ForegroundColor Green
    Write-Host "BENCHMARK COMPLETE!" -ForegroundColor Green
    Write-Host $separator -ForegroundColor Green
    Write-Host ""
    Write-Host "Results saved to: benchmarks\results\" -ForegroundColor Cyan
    Write-Host "Summary report: benchmarks\results\BENCHMARK_SUMMARY.md" -ForegroundColor Cyan
}
