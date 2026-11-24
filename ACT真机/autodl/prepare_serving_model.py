"""Create an audited ACT serving copy without changing the source checkpoint."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from image_preprocessing import CONTRACT_FILENAME, validate_model_contract


MANIFEST_NAME = ".act_serving_manifest.json"
OVERRIDES = {
    "pretrained_backbone_weights": None,
    "push_to_hub": False,
    "repo_id": None,
}


def file_inventory(directory):
    files = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.relative_to(directory).as_posix()):
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise ValueError(f"Model contains a link outside the supported regular-file layout: {path}")
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files.append({
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            })
    return files


def read_source_config(model):
    if not model.is_dir():
        raise ValueError(f"Model directory does not exist: {model}")
    if (model / MANIFEST_NAME).exists():
        raise ValueError("Use the source checkpoint, not an existing serving copy.")
    config = json.loads((model / "config.json").read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or config.get("type") != "act":
        raise ValueError("The source checkpoint must have an ACT config.")
    if (model / CONTRACT_FILENAME).exists():
        validate_model_contract(model)
    for filename in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        path = model / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Required model file is missing or empty: {filename}")
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor = json.loads((model / filename).read_text(encoding="utf-8-sig"))
        if not isinstance(processor, dict) or not isinstance(processor.get("steps"), list):
            raise ValueError(f"Invalid processor configuration: {filename}")
        for step in processor["steps"]:
            if not isinstance(step, dict):
                raise ValueError(f"Invalid processor step in {filename}")
            state_file = step.get("state_file")
            if state_file is not None:
                if not isinstance(state_file, str) or not state_file:
                    raise ValueError(f"Invalid processor state file in {filename}")
                state_path = model / state_file
                if (Path(state_file).is_absolute() or not state_path.resolve().is_relative_to(model)
                        or not state_path.is_file() or state_path.stat().st_size == 0):
                    raise ValueError(f"Missing or invalid processor state file: {state_file}")
    return config


def verify_serving_copy(directory, expected_manifest):
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid serving provenance: {directory}")
    for key, expected in expected_manifest.items():
        if key != "source_model_directory" and manifest.get(key) != expected:
            raise ValueError(f"Serving provenance differs for {directory}: {key}")
    if not isinstance(manifest.get("source_model_directory"), str):
        raise ValueError("Serving copy has no source directory provenance.")
    actual = [entry for entry in file_inventory(directory) if entry["path"] != MANIFEST_NAME]
    if actual != expected_manifest["serving_files"]:
        raise ValueError(f"Serving file hashes differ: {directory}")


def publish_without_replace(staging, destination):
    if os.name == "nt":
        # Windows rename fails if the destination already exists.
        staging.rename(destination)
        return
    if sys.platform != "linux":
        raise RuntimeError("Serving preparation supports Linux and Windows.")
    # AutoDL runs Linux. RENAME_NOREPLACE keeps even an existing empty directory intact.
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


def prepare_model(model, output_root):
    model = model.resolve()
    output_root = output_root.resolve()
    if output_root.is_relative_to(model):
        raise ValueError("The serving output root must be outside the source model directory.")
    config = read_source_config(model)
    source_files = file_inventory(model)
    encoded_inventory = json.dumps(source_files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    source_digest = hashlib.sha256(encoded_inventory.encode("utf-8")).hexdigest()
    config.update(OVERRIDES)
    config_bytes = (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    serving_files = [dict(entry) for entry in source_files]
    config_entry = next(entry for entry in serving_files if entry["path"] == "config.json")
    config_entry.update(size_bytes=len(config_bytes), sha256=hashlib.sha256(config_bytes).hexdigest())
    weights_digest = next(entry["sha256"] for entry in source_files if entry["path"] == "model.safetensors")
    manifest = {
        "schema_version": 1,
        "source_model_directory": str(model),
        "source_manifest_sha256": source_digest,
        "source_model_weights_sha256": weights_digest,
        "source_files": source_files,
        "serving_config_overrides": OVERRIDES,
        "serving_files": serving_files,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / source_digest
    if destination.exists():
        verify_serving_copy(destination, manifest)
        return destination

    staging = Path(tempfile.mkdtemp(prefix=".act-serving-staging-", dir=output_root)).resolve()
    try:
        shutil.copytree(model, staging, dirs_exist_ok=True, symlinks=True)
        if file_inventory(staging) != source_files:
            raise ValueError("Copied source files differ from the initial checkpoint inventory.")
        (staging / "config.json").write_bytes(config_bytes)
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        if file_inventory(model) != source_files:
            raise ValueError("Source checkpoint changed while preparing the serving copy.")
        verify_serving_copy(staging, manifest)
        try:
            publish_without_replace(staging, destination)
        except FileExistsError:
            # A concurrent successful preparation can publish the same content first.
            verify_serving_copy(destination, manifest)
        verify_serving_copy(destination, manifest)
        return destination
    finally:
        if staging.exists():
            if staging.resolve().parent != output_root or not staging.name.startswith(".act-serving-staging-"):
                raise RuntimeError("Refusing to clean a staging directory outside the output root.")
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        destination = prepare_model(args.model, args.output_root)
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration) as error:
        print(f"Serving model preparation failed: {error}", file=sys.stderr)
        return 1
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
