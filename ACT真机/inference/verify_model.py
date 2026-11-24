"""Verify the extracted 8,000-step inference checkpoint against the release manifest."""

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_PATH = Path(__file__).with_name("model-manifest.json")


def verify_model(model: Path) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for name, identity in manifest["files"].items():
        path = model / name
        if path.stat().st_size != identity["bytes"]:
            raise ValueError(f"Checkpoint length differs: {name}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != identity["sha256"]:
                raise ValueError(f"Checkpoint SHA-256 differs: {name}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Extracted pretrained_model directory.")
    checkpoint = verify_model(parser.parse_args().model)
    print(json.dumps({
        "status": "verified",
        "step": checkpoint["checkpoint_step"],
        "verified_files": len(checkpoint["files"]),
        "weights_sha256": checkpoint["source_weights_sha256"],
    }, indent=2))
