# Start the Paramesh AI web UI (FastAPI + Three.js viewer).
# Default parameters: host 127.0.0.1, port 8080.
# Usage:  .\start-server.ps1
# Press Ctrl+C to stop the server cleanly.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"

# Free port 8080 if a previous instance is still bound to it.
$existing = (Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue).OwningProcess
if ($existing) {
    Write-Host "Stopping existing process on port 8080 (PID $existing)..."
    Stop-Process -Id $existing -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

Write-Host "Starting Paramesh AI server at http://127.0.0.1:8080 ..."
Write-Host "Press Ctrl+C to stop."

# Run python as a tracked child sharing this console (so Ctrl+C reaches it),
# and force-kill it in the finally block in case the signal doesn't
# propagate — this guarantees the port is freed when the script ends.
$proc = Start-Process -FilePath "python" `
    -ArgumentList "-m", "mesh2param.viz_server.app" `
    -NoNewWindow -PassThru
try {
    Wait-Process -Id $proc.Id
} finally {
    if (-not $proc.HasExited) {
        Write-Host "`nShutting down server (PID $($proc.Id))..."
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
}
