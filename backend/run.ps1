# DriveGuard AI backend launcher (Windows PowerShell).
#
# NOTE: kept ASCII-only on purpose. Windows PowerShell 5.1 reads .ps1 files using the
# system ANSI code page (GBK on zh-CN), so non-ASCII text breaks parsing.
#
#   .\run.ps1                    # default port 8000, token = dev-token
#   .\run.ps1 -Port 8123         # custom port
#   .\run.ps1 -AdminToken ""     # run without auth (local read-only demo)
#
# Interpreter version comes from .python-version (override with -PythonVersion).
# Dependencies come from requirements-dev.lock.txt, which pins the versions the
# backend test suite passes with; requirements.txt keeps the readable ranges.

param(
    [int]$Port = 8000,
    [string]$AdminToken = "dev-token",
    [string]$PythonVersion = ""
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not $PythonVersion) {
    $versionFile = Join-Path $PSScriptRoot ".python-version"
    if (Test-Path $versionFile) {
        $PythonVersion = (Get-Content $versionFile -Raw).Trim()
    }
    else {
        $PythonVersion = "3.11"
    }
}

$lock = Join-Path $PSScriptRoot "requirements-dev.lock.txt"
if (-not (Test-Path $lock)) {
    Write-Host "[error] requirements-dev.lock.txt is missing." -ForegroundColor Red
    exit 1
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "[setup] creating virtualenv .venv with Python $PythonVersion ..." -ForegroundColor Cyan
    py "-$PythonVersion" -m venv .venv
    if (-not (Test-Path $python)) {
        Write-Host "[error] Python $PythonVersion not found. Install it or create .venv manually." -ForegroundColor Red
        exit 1
    }
}

$actual = "$(& $python -c "import sys; print('%d.%d' % sys.version_info[:2])")".Trim()
if ($actual -ne $PythonVersion) {
    Write-Host "[error] .venv runs Python $actual but .python-version requires $PythonVersion." -ForegroundColor Red
    Write-Host "        Delete .venv and re-run, or pass -PythonVersion $actual to keep this environment." -ForegroundColor Red
    exit 1
}

$stamp = "$PythonVersion|$((Get-FileHash -Path $lock -Algorithm SHA256).Hash)"
$sentinel = Join-Path $PSScriptRoot ".venv\.deps-installed"
$installed = ""
if (Test-Path $sentinel) {
    $installed = "$(Get-Content $sentinel -Raw)".Trim()
}
if ($installed -ne $stamp) {
    Write-Host "[setup] installing pinned dependencies (first run may take a while) ..." -ForegroundColor Cyan
    & $python -m pip install --upgrade pip --quiet
    & $python -m pip install -r $lock --quiet
    Set-Content -Path $sentinel -Value $stamp -Encoding ascii
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
