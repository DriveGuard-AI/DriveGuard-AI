# DriveGuard AI - stop local dev servers (Windows PowerShell).
#
# Stops whatever is LISTENING on the given ports, plus a matching uvloop/vite parent
# wrapper process if present. Nothing else is touched.
#
# NOTE: kept ASCII-only on purpose -- Windows PowerShell 5.1 reads .ps1 files with the
# system ANSI code page (GBK on zh-CN), so non-ASCII text breaks parsing.
#
#   .\stop.ps1                 # stop backend (8000) and vite dev server (5173)
#   .\stop.ps1 -Ports 8000     # stop only the backend
#   .\stop.ps1 -WhatIf         # only report what would be stopped

param(
    [int[]]$Ports = @(8000, 5173),
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$stopped = @()

foreach ($port in $Ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if (-not $connections) {
        Write-Host "[stop] port $port : nothing listening" -ForegroundColor DarkGray
        continue
    }

    foreach ($target in ($connections | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique)) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$target" -ErrorAction SilentlyContinue
        if (-not $proc) { continue }

        Write-Host "[stop] port $port -> PID $target ($($proc.Name))" -ForegroundColor Yellow

        if ($WhatIf) {
            Write-Host "       (whatif) would run: taskkill /PID $target /T /F" -ForegroundColor DarkGray
            continue
        }

        & taskkill /PID $target /T /F | Out-Null
        $stopped += $target

        # Some launchers keep a wrapper parent alive (e.g. venv python -> base python).
        $parentId = $proc.ParentProcessId
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$parentId" -ErrorAction SilentlyContinue
        if ($parent -and $parent.CommandLine -match 'uvicorn|vite|http\.server') {
            Write-Host "[stop]   parent PID $parentId also matches, stopping it" -ForegroundColor Yellow
            & taskkill /PID $parentId /T /F | Out-Null
            $stopped += $parentId
        }
    }
}

if (-not $WhatIf) {
    if ($stopped.Count -eq 0) {
        Write-Host "[stop] nothing to stop" -ForegroundColor Cyan
    } else {
        Write-Host "[stop] stopped PID(s): $($stopped -join ', ')" -ForegroundColor Green
    }
}
