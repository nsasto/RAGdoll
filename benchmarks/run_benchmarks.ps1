# RAGdoll Benchmark Runner
# Automates benchmark execution and comparison

param(
    [int]$Subset = 51,
    [string]$Dataset = "2wikimultihopqa",
    [switch]$CreateOnly,
    [switch]$BenchmarkOnly,
    [switch]$SkipBaseline,
    [switch]$IncludeNoChunking,
    [string]$Mode  # Specific mode: "vector", "pagerank", "hybrid", or empty for all modes
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
if ($Mode) {
    Write-Host "Mode:    $Mode (single mode)" -ForegroundColor Green
}
else {
    Write-Host "Mode:    All modes (vector, pagerank, hybrid)" -ForegroundColor Green
}
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

# Determine which modes to run
$modes = if ($Mode) {
    @($Mode)
}
else {
    @("vector", "pagerank", "hybrid")
}

# Run RAGdoll modes (chunked)
$sectionNumber = 2
foreach ($ragdollMode in $modes) {
    $modeTitle = switch ($ragdollMode) {
        "vector" { "RAGdoll (Vector Only - Chunked)" }
        "pagerank" { "RAGdoll (PageRank Graph - Chunked)" }
        "hybrid" { "RAGdoll (Hybrid: Vector + Graph - Chunked)" }
    }
    
    Write-Host $separator -ForegroundColor Cyan
    Write-Host "$sectionNumber. $modeTitle" -ForegroundColor Cyan
    Write-Host $separator -ForegroundColor Cyan
    Write-Host ""
    
    if ($CreateOnly -or -not $BenchmarkOnly) {
        Write-Host "Creating $ragdollMode index..." -ForegroundColor Green
        python ragdoll_benchmark.py -d $Dataset -n $Subset --mode $ragdollMode --create
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to create $ragdollMode index" -ForegroundColor Red
            exit 1
        }
    }
    
    if ($BenchmarkOnly -or -not $CreateOnly) {
        Write-Host "Running benchmark..." -ForegroundColor Green
        python ragdoll_benchmark.py -d $Dataset -n $Subset --mode $ragdollMode --benchmark
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to benchmark $ragdollMode" -ForegroundColor Red
            exit 1
        }
    }
    
    Write-Host ""
    $sectionNumber++
}

# Run RAGdoll modes with no-chunking if requested
if ($IncludeNoChunking) {
    foreach ($ragdollMode in $modes) {
        $modeTitle = switch ($ragdollMode) {
            "vector" { "RAGdoll (Vector Only - No Chunking)" }
            "pagerank" { "RAGdoll (PageRank Graph - No Chunking)" }
            "hybrid" { "RAGdoll (Hybrid: Vector + Graph - No Chunking)" }
        }
        
        Write-Host $separator -ForegroundColor Cyan
        Write-Host "$sectionNumber. $modeTitle" -ForegroundColor Cyan
        Write-Host $separator -ForegroundColor Cyan
        Write-Host ""
        
        if ($CreateOnly -or -not $BenchmarkOnly) {
            Write-Host "Creating $ragdollMode no-chunking index..." -ForegroundColor Green
            python ragdoll_benchmark.py -d $Dataset -n $Subset --mode $ragdollMode --no-chunking --create
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "ERROR: Failed to create $ragdollMode no-chunking index" -ForegroundColor Red
                exit 1
            }
        }
        
        if ($BenchmarkOnly -or -not $CreateOnly) {
            Write-Host "Running benchmark..." -ForegroundColor Green
            python ragdoll_benchmark.py -d $Dataset -n $Subset --mode $ragdollMode --no-chunking --benchmark
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "ERROR: Failed to benchmark $ragdollMode no-chunking" -ForegroundColor Red
                exit 1
            }
        }
        
        Write-Host ""
        $sectionNumber++
    }
}

Write-Host ""

# Generate comparison report
if ($BenchmarkOnly -or -not $CreateOnly) {
    Write-Host $separator -ForegroundColor Cyan
    Write-Host "$sectionNumber. Generating Comparison Report" -ForegroundColor Cyan
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
