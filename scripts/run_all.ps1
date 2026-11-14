# Full experiment suite. Requires Docker Desktop running and ANTHROPIC_API_KEY in .env
# (or pass -Offline to use the rule planner + regex extractor with no API key).
#   -Group A   main run + load test + budget-matched controls   (main HydraDB, port 8443)
#   -Group B   baselines and ablations                          (dev HydraDB, port 18443)
#   -Group All everything sequentially on the main HydraDB (default)
# A and B can run at the same time in two terminals; run `python -m eval.plot` when both finish.
param([switch]$Offline, [int]$Rounds = 6, [ValidateSet("A", "B", "All")][string]$Group = "All",
      [string[]]$Only = @())   # run just these variants (Group B / All), e.g. -Only heal_only,no_merge

$ErrorActionPreference = "Continue"   # native tools (docker, HF) write progress to stderr
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$py = ".\.venv\Scripts\python.exe"
$off = @(); if ($Offline) { $off = @("--offline") }
$suffix = ""; if ($Offline) { $suffix = "_offline" }

function Run-Variants($names) {
    # -File passes "a,b" as one string, so split it here
    $want = @($Only | ForEach-Object { $_ -split "," } | Where-Object { $_ })
    if ($want.Count -gt 0) { $names = $names | Where-Object { $want -contains $_ } }
    foreach ($v in $names) {
        Write-Host "=== $v ==="
        & $py -u -m eval.run_curve --variant $v --rounds $Rounds @off
    }
}

if ($Group -eq "B") {
    $env:HYDRA_URL = "http://127.0.0.1:18443"; $env:HYDRA_SERVICE = "hydradb-dev"
    docker compose --profile dev up -d hydradb-dev 2>&1 | Out-Null
    Run-Variants @("static", "full_kb", "heal_only", "plans_only", "no_merge")
    if (-not $Offline) { & $py -m eval.baselines --which vector llm }
    exit
}

docker compose up -d hydradb 2>&1 | Out-Null
if (-not (Test-Path "data\prepared\workload.json")) { & $py -m desirepath.data }

# main run (+ query-time repair), then load-test its learned graph before anything resets it
Write-Host "=== desirepath ==="
& $py -u -m eval.run_curve --variant desirepath --rounds $Rounds --serve-repair @off
& $py -m eval.load_test --readers 32 --seconds 60 --rows "results\desirepath$suffix.heldout_r$Rounds.json"
Run-Variants @("random_heal", "noisy_fb")   # random_heal matches desirepath's reading budget

if ($Group -eq "All") {
    Run-Variants @("static", "full_kb", "heal_only", "plans_only", "no_merge")
    if (-not $Offline) { & $py -m eval.baselines --which vector llm }
    & $py -m eval.plot --suffix $suffix
}
