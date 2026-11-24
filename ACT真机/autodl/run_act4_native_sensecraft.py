"""Run the published original ACT4 8,000-step manual profile on SenseCraft."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


PROJECT = Path(__file__).resolve().parents[1]
PROFILE = PROJECT / "inference/trials_profile_arm165_te001_manual.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require_idle_can(port):
    import ctypes

    # This installation maps can0 to PCAN_USBBUS1. Query the adapter without
    # initializing a second CAN connection or sending motor commands.
    if port != "can0":
        raise ValueError("The saved-pose ACT4 setup expects the existing can0 adapter.")
    channel = 0x51  # PCAN_USBBUS1
    condition_parameter = 0x0D  # PCAN_CHANNEL_CONDITION
    available = 0x01  # PCAN_CHANNEL_AVAILABLE
    read_condition = ctypes.WinDLL("PCANBasic.dll").CAN_GetValue
    read_condition.argtypes = (ctypes.c_ushort, ctypes.c_ubyte, ctypes.c_void_p, ctypes.c_uint)
    read_condition.restype = ctypes.c_uint
    condition = ctypes.c_uint()
    result = read_condition(
        channel, condition_parameter, ctypes.byref(condition), ctypes.sizeof(condition),
    )
    if result != 0 or condition.value != available:
        raise RuntimeError(f"can0 is not free for a new client (PCAN status {result:#x}/{condition.value:#x}).")


def prepare(site_path):
    site = read_json(site_path)

    def site_directory(name):
        return (site_path.parent / Path(site[name]).expanduser()).resolve(strict=True)

    runtime = site_directory("runtime_root")
    backend = site_directory("backend_root")
    data_root = site_directory("data_root")
    model = site_directory("model_dir")
    project_id = site["project_id"]
    recording_id = site["recording_id"]
    for identity in (project_id, recording_id):
        if not isinstance(identity, str) or not identity or identity in {".", ".."} or Path(identity).name != identity:
            raise ValueError("Project and recording identifiers must be single directory names.")

    for relative, expected in read_json(PROJECT / "inference/vendor-source-hashes.json").items():
        with (backend / relative).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"SenseCraft source differs from the reviewed runtime: {relative}")

    overlay = PROJECT / "sensecraft_runtime_overlay/py313"
    os.environ["ACT_SENSECRAFT_BACKEND_ROOT"] = str(backend)
    imports = [str(overlay), str(backend), str(PROJECT / "inference")]
    if site.get("pillow_root"):
        imports.insert(0, str(site_directory("pillow_root")))
    sys.path[:0] = imports
    child_paths = [*imports, str(PROJECT / "autodl"), os.environ.get("PYTHONPATH", "")]
    os.environ["PYTHONPATH"] = os.pathsep.join(path for path in child_paths if path)
    from backend.state import StateRepository
    from inference_trials import load_profile, load_zero, robot_identity, valid_capture_request
    from sensecraft_224_adapter import preflight
    from verify_model import verify_model

    repo = StateRepository(data_root)
    active = repo.load_active_project_id()
    setup = repo.load_project_setup(active)
    if active != project_id or not setup or not setup.get("completed"):
        raise ValueError("Select the site configuration's project with completed setup in SenseCraft.")
    recording_path = data_root / "projects" / active / "recordings" / recording_id / "state.json"
    recording = read_json(recording_path)
    config_fields = ("robot_type", "robot_port", "robot_id", "fps", "camera_wrist_role")
    camera_fields = tuple(
        f"camera_{role}_{field}"
        for role in ("front", "wrist")
        for field in ("index", "width", "height", "fps")
    )
    if any(setup.get(key) != recording["setup"].get(key) for key in config_fields + camera_fields):
        raise ValueError("Current SenseCraft setup differs from the recorded ACT4 configuration.")

    checkpoint = verify_model(model)
    args = SimpleNamespace(
        backend_root=backend,
        recording_state=recording_path,
        contract_model_path=model,
        remote_model_path=site["serving_path"],
        server_address=site["server_address"],
        task=recording["task_text"],
        actions_per_chunk=50,
        chunk_size_threshold=0.5,
        aggregate_fn_name="weighted_average",
        history_observation=False,
    )
    profile = load_profile(PROFILE)
    processing = {"arm_transition_rate_deg_s": 165, "temporal_ensemble_coeff": 0.01,
                  "temporal_ensemble_scope": "arm6"}
    if not (profile["enabled"] and profile["operator_controlled"] and profile["manual_advance"]
            and profile["operator_saved_pose"] and profile["max_trials"] is None
            and profile["action_processing"] == processing) or valid_capture_request(profile):
        raise ValueError("Use the published manual profile and the already verified saved pose.")
    calibration = data_root / "calibration/lerobot/robots" / setup["robot_type"]
    report, contract, recorded_setup = preflight(
        args, runtime_root=runtime, calibration_dir=calibration, trial_profile_path=PROFILE,
    )
    zero = load_zero(profile, robot_identity(report))
    report.update(
        checkpoint_step=checkpoint["checkpoint_step"],
        checkpoint_identity="low8000",
        model_weights_sha256=checkpoint["source_weights_sha256"],
        saved_zero_verified=True,
        saved_zero_source=zero["source"],
        retry_hotkey="Ctrl+C",
        finish_hotkey="Ctrl+V",
    )
    return args, report, contract, recorded_setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--low8000", action="store_true", help="The released checkpoint (also the default).")
    parser.add_argument("--arm165-te001-manual", action="store_true",
                        help="The released manual profile (also the default).")
    parser.add_argument("--start", action="store_true", help="Wait for the operator after a successful preflight.")
    options = parser.parse_args()
    args, report, contract, setup = prepare(options.site.resolve(strict=True))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if options.start:
        from sensecraft_224_adapter import run_client

        require_idle_can(report["robot_port"])
        os.environ["ACT_TRIAL_VIDEO"] = "1"
        os.environ["ACT_INFERENCE_TRIALS_PROFILE"] = str(PROFILE)
        os.environ["ACT_FEEDBACK_ROOT"] = str(PROJECT / "local/feedback")
        os.environ["WEBUI_RUN_TRACE_TAG"] = "act4_low8000_arm165_te001_manual"
        run_client(args, report, contract, setup)


if __name__ == "__main__":
    main()
