# Start the Paramesh AI web UI (FastAPI + Three.js viewer).
# Default parameters: host 127.0.0.1, port 8080.
# Usage:  .\start-server.ps1
# Press Ctrl+C to stop the server cleanly.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"

# Preferred port 8080; if it's busy we first try to evict a stale
# previous instance, and if something still holds it (another app),
# fall back to the next free port in 8081..8090.
# Use netstat instead of Get-NetTCPConnection — Get-NetTCPConnection
# sometimes misses zombie listeners that another snapshot interface
# (netstat) still sees, and the server then fails to bind with
# WSAEADDRINUSE even though Get-NetTCPConnection reported nothing.
function Get-PortListeners([int]$Port) {
    netstat -ano | Select-String (":" + $Port + "\s+\S+\s+LISTENING\s+(\d+)") |
        ForEach-Object { [int]$_.Matches[0].Groups[1].Value } | Select-Object -Unique
}

$port = 8080
$pids = @(Get-PortListeners $port)
foreach ($p in $pids) {
    Write-Host "Stopping existing process on port $port (PID $p)..."
    Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
}
if ($pids.Count -gt 0) {
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 500
        if (-not (Get-PortListeners $port)) { break }
    }
}

# Still busy (another application owns it)? Pick the next free port.
if (Get-PortListeners $port) {
    foreach ($candidate in 8081..8090) {
        if (-not (Get-PortListeners $candidate)) {
            Write-Host "Port 8080 is in use by another application; using port $candidate instead." -ForegroundColor Yellow
            $port = $candidate
            break
        }
    }
    if (Get-PortListeners $port) {
        Write-Host "ERROR: no free port found in 8080-8090." -ForegroundColor Red
        exit 1
    }
}

$env:PARAMESH_PORT = "$port"
Write-Host "Starting Paramesh AI server at http://127.0.0.1:$port ..."
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
