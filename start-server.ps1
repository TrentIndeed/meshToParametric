# Start the Paramesh AI web UI (FastAPI + Three.js viewer).
# Default parameters: host 127.0.0.1, port 8080.
# Usage:  .\start-server.ps1
# Press Ctrl+C to stop the server cleanly.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"

# Free port 8080 if a previous instance is still bound to it.
# Use netstat instead of Get-NetTCPConnection — Get-NetTCPConnection
# sometimes misses zombie listeners that another snapshot interface
# (netstat) still sees, and the server then fails to bind with WSAEADDRINUSE
# even though Get-NetTCPConnection reported nothing.
$listening = netstat -ano | Select-String ":8080\s+0\.0\.0\.0:0\s+LISTENING\s+(\d+)"
foreach ($m in $listening) {
    $pidStr = $m.Matches[0].Groups[1].Value
    Write-Host "Stopping existing process on port 8080 (PID $pidStr)..."
    Stop-Process -Id ([int]$pidStr) -Force -ErrorAction SilentlyContinue
}
if ($listening) { Start-Sleep -Seconds 1 }

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
