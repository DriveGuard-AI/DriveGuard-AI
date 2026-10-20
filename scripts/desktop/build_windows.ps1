#requires -Version 5.1
<#
.SYNOPSIS
  Build the DriveGuard AI Windows x64 desktop installer.

.DESCRIPTION
  Requires Python 3.11/3.12, Node 22+, and the built-in model service values:
    $env:DGB_MODEL_SERVICE_URL = "https://dms.example.com"
    $env:DGB_MODEL_SERVICE_TOKEN = "<at least 24 characters>"
  Outputs to desktop/release: NSIS installer, portable exe, SHA256SUMS.txt.

  Keep this file ASCII-only: Windows PowerShell 5.1 reads .ps1 using the
  system ANSI code page, so non-ASCII text can be garbled.
#>
$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

if (-not $env:DGB_MODEL_SERVICE_URL) { throw "DGB_MODEL_SERVICE_URL is required" }
if (-not $env:DGB_MODEL_SERVICE_TOKEN) { throw "DGB_MODEL_SERVICE_TOKEN is required" }
if ($env:DGB_MODEL_SERVICE_TOKEN.Length -lt 24) { throw "DGB_MODEL_SERVICE_TOKEN must be at least 24 characters" }

Write-Host "[1/6] Build frontend"
Push-Location frontend
npm ci
if ($LASTEXITCODE -ne 0) { throw "frontend npm ci failed" }
npm run build
if ($LASTEXITCODE -ne 0) { throw "frontend build failed" }
Pop-Location

Write-Host "[2/6] Install backend packaging dependencies"
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt "pyinstaller>=6.10,<7"
if ($LASTEXITCODE -ne 0) { throw "backend dependency install failed" }

Write-Host "[3/6] Write embedded config"
if (Test-Path desktop/dist) { Remove-Item desktop/dist -Recurse -Force }
New-Item -ItemType Directory -Force desktop/dist | Out-Null
python scripts/desktop/write_config.py --output desktop/dist/config.json
if ($LASTEXITCODE -ne 0) { throw "config generation failed" }

Write-Host "[4/6] Package backend with PyInstaller"
python -m PyInstaller desktop/packaging/backend.spec --noconfirm `
  --distpath desktop/dist --workpath desktop/build/pyinstaller
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

Write-Host "[5/6] Copy frontend and build Electron installer"
New-Item -ItemType Directory -Force desktop/dist/frontend | Out-Null
Copy-Item frontend/dist/* desktop/dist/frontend -Recurse -Force
Push-Location desktop
if (Test-Path package-lock.json) { npm ci } else { npm install }
if ($LASTEXITCODE -ne 0) { throw "desktop npm install failed" }
$env:CSC_IDENTITY_AUTO_DISCOVERY = "false"
npm run dist
if ($LASTEXITCODE -ne 0) { throw "electron-builder failed" }
Pop-Location

Write-Host "[6/7] Verify unpacked app and generate SHA256SUMS"
$Release = Join-Path $Root "desktop\release"
$Unpacked = Join-Path $Release "win-unpacked"
$AppExe = Join-Path $Unpacked "DriveGuard AI.exe"
$BackendExe = Join-Path $Unpacked "resources\backend\driveguard-backend.exe"

if (-not (Test-Path $AppExe)) {
  throw "Main executable missing from win-unpacked: $AppExe"
}
if (-not (Test-Path $BackendExe)) {
  throw "Backend executable missing from win-unpacked: $BackendExe"
}

$AppExeInfo = Get-Item $AppExe
Write-Host "Main executable: $($AppExeInfo.FullName) ($([math]::Round($AppExeInfo.Length / 1MB, 1)) MB)"

# Electron 34 x64 主程序约 180 MB；明显小于 100 MB 说明复制/打包被截断。
if ($AppExeInfo.Length -lt 100MB) {
  throw "Main executable is only $([math]::Round($AppExeInfo.Length / 1MB, 1)) MB; expected about 182 MB. Aborting to avoid publishing a truncated build."
}

$ExeFiles = Get-ChildItem $Release -Filter *.exe -File
if (-not $ExeFiles) { throw "No release exe found" }
$Lines = foreach ($File in $ExeFiles) {
  $Hash = (Get-FileHash $File.FullName -Algorithm SHA256).Hash.ToLower()
  "$Hash  $($File.Name)"
}
$AppHash = (Get-FileHash $AppExe -Algorithm SHA256).Hash.ToLower()
$Lines += "$AppHash  win-unpacked/DriveGuard AI.exe"
$Lines | Set-Content (Join-Path $Release "SHA256SUMS.txt") -Encoding ASCII
Get-Content (Join-Path $Release "SHA256SUMS.txt")

Write-Host "[7/7] Create unpacked diagnostic zip"
$Version = (Get-Content (Join-Path $Root "desktop\package.json") -Raw | ConvertFrom-Json).version
$Zip = Join-Path $Release "DriveGuardAI-$Version-win-unpacked.zip"
Compress-Archive -Path "$Unpacked\*" -DestinationPath $Zip -Force
Write-Host "Unpacked zip: $Zip"

Write-Host "Done. Artifacts: $Release"
