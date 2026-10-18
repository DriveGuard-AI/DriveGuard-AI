# DriveGuard AI backend launcher (Windows PowerShell).
#
# NOTE: kept ASCII-only on purpose. Windows PowerShell 5.1 reads .ps1 files using the
# system ANSI code page (GBK on zh-CN), so non-ASCII text breaks parsing.
#
#   .\run.ps1                    # default port 8000, token = dev-token
#   .\run.ps1 -Port 8123         # custom port
#   .\run.ps1 -AdminToken ""     # run without auth (local read-only demo)

param(
    [int]$Port = 8000,
    [string]$AdminToken = "dev-token"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "[setup] creating virtualenv .venv with Python 3.11 ..." -ForegroundColor Cyan
    py -3.11 -m venv .venv
    if (-not (Test-Path $python)) {
        Write-Host "[error] Python 3.11 not found. Install it or create .venv manually." -ForegroundColor Red
        exit 1
    }
}

$sentinel = Join-Path $PSScriptRoot ".venv\.deps-installed"
if (-not (Test-Path $sentinel)) {
    Write-Host "[setup] installing dependencies (first run may take a while) ..." -ForegroundColor Cyan
    & $python -m pip install --upgrade pip --quiet
    & $python -m pip install -r requirements-dev.txt --quiet
    New-Item -ItemType File -Force -Path $sentinel | Out-Null
}

$env:DGB_ADMIN_TOKEN = $AdminToken

if ($AdminToken) {
    $tokenInfo = $AdminToken
} else {
    $tokenInfo = "(disabled - anonymous read-only)"
}

Write-Host "[run] dashboard http://127.0.0.1:$Port/    openapi http://127.0.0.1:$Port/docs" -ForegroundColor Green
Write-Host "[run] admin token: $tokenInfo" -ForegroundColor Green

& $python -m uvicorn app.main:app --host 127.0.0.1 --port $Port
