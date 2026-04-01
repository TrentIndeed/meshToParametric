"""
Deploy mesh2param to the existing Hetzner VPS (alongside operatorDashboard).

Uses the same SSH + Docker + Caddy pattern from operatorDashboard.
mesh2param runs on port 8081 (operator uses 8000/3000).

Usage:
    python deploy.py                          # deploy to VPS_IP from .env
    python deploy.py --ip 1.2.3.4             # deploy to specific IP
    python deploy.py --domain mesh2param.com  # configure HTTPS via Caddy

Requires:
    - SSH key at ~/.ssh/id_ed25519 (same as operatorDashboard)
    - VPS already provisioned with Docker + Caddy (operatorDashboard's cloud-init)
"""

import argparse
import asyncio
import os
import subprocess
import sys

from dotenv import load_dotenv
load_dotenv()

_GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
_REPO_BASE = "github.com/TrentIndeed/meshToParametric.git"
DEPLOY_REPO = f"https://{_GITHUB_TOKEN}@{_REPO_BASE}" if _GITHUB_TOKEN else f"https://{_REPO_BASE}"
DEPLOY_BRANCH = os.getenv("DEPLOY_BRANCH", "main")
SSH_KEY_PATH = os.getenv("SSH_KEY_PATH", os.path.expanduser("~/.ssh/id_ed25519"))
VPS_IP = os.getenv("VPS_IP", "")


async def deploy(ip: str, domain: str = "", auth_secret: str = ""):
    """Deploy mesh2param to the VPS."""
    if not auth_secret:
        auth_secret = os.getenv("AUTH_SECRET", "mesh2param-default-secret-change-me")

    # Caddy config block — appends to existing Caddyfile
    caddy_block = ""
    if domain:
        caddy_block = f"""
# Append mesh2param to Caddy config
# (Don't overwrite — operatorDashboard's config is already there)
cat >> /etc/caddy/Caddyfile << 'CADDY'

{domain} {{
    reverse_proxy localhost:8081
}}
CADDY
systemctl reload caddy
"""

    deploy_script = f"""#!/bin/bash
set -e
echo "=== Deploying mesh2param ==="

# Clone or pull
if [ -d /opt/meshToParametric ]; then
    cd /opt/meshToParametric && git pull origin {DEPLOY_BRANCH}
else
    cd /opt && git clone -b {DEPLOY_BRANCH} {DEPLOY_REPO}
fi

cd /opt/meshToParametric

# Write .env
cat > .env << 'ENVEOF'
AUTH_SECRET={auth_secret}
DATABASE_URL=sqlite:////app/data/mesh2param.db
ENVEOF

# Build and deploy (port 8081, doesn't conflict with operator on 8000/3000)
docker compose up -d --build

{caddy_block}

echo "DEPLOY_COMPLETE"
"""

    print(f"Deploying to {ip}...")
    result = await _ssh_exec(ip, deploy_script)

    if "DEPLOY_COMPLETE" in result:
        url = f"https://{domain}" if domain else f"http://{ip}:8081"
        print(f"\nDeployed successfully!")
        print(f"  App:     {url}")
        print(f"  Landing: {url}/landing")
        print(f"  API:     {url}/api/status")
    else:
        print(f"\nDeploy may have failed:\n{result[-500:]}")


async def _ssh_exec(ip: str, script: str) -> str:
    """Execute a script on the remote server via SSH."""
    proc = await asyncio.create_subprocess_exec(
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=30",
        "-i", SSH_KEY_PATH,
        f"root@{ip}",
        "bash", "-s",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(
        proc.communicate(script.encode()),
        timeout=600,  # 10 min for Docker build
    )
    output = stdout.decode() + stderr.decode()
    print(output[-2000:])  # Show tail of output
    if proc.returncode != 0:
        raise RuntimeError(f"SSH failed (exit {proc.returncode})")
    return output


def main():
    parser = argparse.ArgumentParser(description="Deploy mesh2param to Hetzner VPS")
    parser.add_argument("--ip", default=VPS_IP, help="VPS IP address (or set VPS_IP in .env)")
    parser.add_argument("--domain", default="parameshai.com", help="Domain for HTTPS")
    parser.add_argument("--auth-secret", default="", help="AUTH_SECRET for production")
    args = parser.parse_args()

    if not args.ip:
        print("Error: No VPS IP. Use --ip or set VPS_IP in .env")
        sys.exit(1)

    asyncio.run(deploy(args.ip, args.domain, args.auth_secret))


if __name__ == "__main__":
    main()
