"""Claude CLI wrapper — uses OAuth via the installed claude CLI.

No API key needed. The CLI handles authentication via the user's
Claude Max/Pro subscription. Costs go to the subscription, not an
API billing account.

Prerequisites:
    npm install -g @anthropic-ai/claude-code
    claude "hello"   # one-time browser auth
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


def find_claude() -> str:
    """Locate the claude CLI binary."""
    found = shutil.which("claude")
    if found:
        return found
    appdata = os.environ.get("APPDATA", "")
    for ext in (".cmd", ".exe", ""):
        p = os.path.join(appdata, "npm", f"claude{ext}")
        if os.path.isfile(p):
            return p
    return "claude"


CLAUDE = find_claude()


def call_ai(prompt: str, model: str = "claude-sonnet-4-6", timeout: int = 180) -> str:
    """Call Claude via the CLI and return the text response.

    Args:
        prompt: The prompt text (passed via stdin).
        model: Model to use.
        timeout: Max seconds to wait.

    Returns:
        The model's text response.

    Raises:
        RuntimeError: If the CLI fails or is not installed.
    """
    try:
        result = subprocess.run(
            [CLAUDE, "--model", model, "--output-format", "json", "--max-turns", "1"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Claude CLI timed out after {timeout}s")

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed (exit {result.returncode}): {result.stderr[:500]}")

    text = result.stdout.strip()
    try:
        data = json.loads(text)
        if "result" in data:
            return data["result"]
    except (json.JSONDecodeError, KeyError):
        pass
    return text


def is_available() -> bool:
    """Check if the Claude CLI is installed and authenticated."""
    try:
        result = subprocess.run(
            [CLAUDE, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False
