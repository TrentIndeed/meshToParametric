# Start the Paramesh AI web UI inside the Linux Docker container.
#
# Why the container: the alpha-wrap watertight repair (ingest) and the exact-IoU
# residual arbiter only work on Linux (pymeshlab's alpha-wrap plugin won't load
# on Windows). Running natively, those are no-ops — AI/scan meshes come out far
# worse. This script builds + runs the same image that deploys to Hetzner.
#
# Usage:
#   .\start-server.ps1            # build (if needed) + run on http://127.0.0.1:8080
#   .\start-server.ps1 -NoBuild   # skip the image build, just (re)start the container
#   .\start-server.ps1 -Native    # old behaviour: native python (NO alpha-wrap)
# Press Ctrl+C to stop (the container is removed on exit).
param([switch]$NoBuild, [switch]$Native)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"

# netstat (not Get-NetTCPConnection) catches zombie listeners the latter misses.
function Get-PortListeners([int]$Port) {
    netstat -ano | Select-String (":" + $Port + "\s+\S+\s+LISTENING\s+(\d+)") |
        ForEach-Object { [int]$_.Matches[0].Groups[1].Value } | Select-Object -Unique
}

# ----- Native fallback (no container; alpha-wrap is a no-op on Windows) --------
if ($Native) {
    $port = 8080
    foreach ($p in @(Get-PortListeners $port)) {
        Write-Host "Stopping existing process on port $port (PID $p)..."
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
    }
    $env:PARAMESH_PORT = "$port"
    Write-Host "Starting Paramesh AI (native, NO alpha-wrap) at http://127.0.0.1:$port ..."
    python -m mesh2param.viz_server.app
    return
}

# ----- Container path ---------------------------------------------------------
$image = "paramesh-server"
$container = "paramesh-server"

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker is not running. Start Docker Desktop, or use -Native." -ForegroundColor Red
    exit 1
}

if (-not $NoBuild) {
    Write-Host "Building $image (Linux image with the alpha-wrap / exact-IoU stack)..." -ForegroundColor Cyan
    docker build -t $image .
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: image build failed." -ForegroundColor Red; exit 1 }
}

# Pick a host port: prefer 8080, evict our own stale container first, else 8081-8090.
docker rm -f $container *> $null
$port = 8080
if (Get-PortListeners $port) {
    foreach ($candidate in 8081..8090) {
        if (-not (Get-PortListeners $candidate)) { $port = $candidate; break }
    }
    if (Get-PortListeners $port) {
        Write-Host "ERROR: no free port in 8080-8090." -ForegroundColor Red; exit 1
    }
    Write-Host "Port 8080 busy; using $port." -ForegroundColor Yellow
}

# Persist the SQLite DB in a NAMED volume (NOT a host bind-mount): SQLite file
# locking deadlocks over Docker Desktop's Windows filesystem mount and hangs
# startup. Serve the host's live source + models so edits and the sample meshes
# show up without a rebuild.
$envFileArg = @()
if (Test-Path "$PSScriptRoot\.env") { $envFileArg = @("--env-file", "$PSScriptRoot\.env") }

Write-Host "Starting $container at http://127.0.0.1:$port  (Ctrl+C to stop)" -ForegroundColor Green
Write-Host "Note: the Claude CLI (AI chat) is unavailable inside the container; it degrades gracefully." -ForegroundColor DarkGray

docker run --rm --name $container `
    -p "${port}:8080" `
    -v "paramesh-data:/app/data" `
    -v "${PSScriptRoot}\mesh2param:/app/mesh2param" `
    -v "${PSScriptRoot}\models:/app/models" `
    @envFileArg `
    -e DATABASE_URL="sqlite:////app/data/mesh2param.db" `
    -e PYTHONUTF8=1 `
    $image
