"""Expose the reviewed SenseCraft payload to opt-in overlay subprocesses.

The external backend already supplies this exact repository environment value.
Only its known immutable payload is added. No backend or device is imported.
"""

import os
from pathlib import Path
import sys


def _expose_reviewed_payload():
    configured = os.environ.get("WEBUI_BACKEND_REPO_ROOT")
    expected = os.environ.get("ACT_SENSECRAFT_BACKEND_ROOT")
    if not configured or not expected:
        return
    try:
        payload = Path(configured).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return
    if payload != Path(expected).resolve():
        return
    if not (payload / "backend/__init__.py").is_file():
        return
    if not (payload / "backend/devices/camera_subprocess.py").is_file():
        return
    if str(payload) not in sys.path:
        sys.path.append(str(payload))


_expose_reviewed_payload()
