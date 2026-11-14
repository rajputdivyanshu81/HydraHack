# Waits for experiments to finish, renders the chart + README table, commits and pushes. Runs unattended.
#   phase 1: main run + load test   ->  push
#   phase 2: remaining ablations    ->  push again
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"

function Finished($name) {
    $p = "results\$name.json"
    if (-not (Test-Path $p)) { return $false }
    try { $d = Get-Content $p -Raw | ConvertFrom-Json } catch { return $false }
    return [bool]($d.PSObject.Properties.Name -contains "finished")
}

function Publish($msg) {
    & $py -m eval.plot 2>&1 | Out-Null
    git add README.md results 2>&1 | Out-Null
    $body = "$msg`n`nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    git commit -q -m $body 2>&1 | Out-Null
    git push origin main 2>&1 | Out-Null
    Write-Host "$(Get-Date -Format HH:mm) pushed: $msg"
}

while (-not ((Finished "desirepath") -and (Test-Path "results\load_test_32.json"))) { Start-Sleep 30 }
Publish "Add main DesirePath results, load test, chart and README results table"

while (-not ((Finished "no_merge") -and (Finished "random_heal") -and (Finished "noisy_fb"))) { Start-Sleep 60 }
Publish "Add no-merge ablation, random-ingestion and wrong-feedback controls"
Write-Host "ALL DONE"
