"""Create or resume a single-variable ACT4 main-LR continuation on AutoDL.

The regular ACT4 entry owns datasets, processors, compilation and the objective.
This entry restores its complete checkpoint before changing the main-group LR.
--check-only hashes assets and builds the command without importing a framework.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import sys


RUN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
METADATA = "lr_branch.json"
ASSET_MANIFEST = "lr_branch_checkpoint_manifest.json"
BACKBONE_LR = 1e-5

# DataLoader spawn runs this entry before decoding, without calling main().
if __name__ == "__mp_main__" and os.environ.get("ACT_VIDEO_CACHE_MANIFEST"):
    import train_with_video_cache


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_hashes(checkpoint):
    return {path.relative_to(checkpoint).as_posix(): file_hash(path)
            for path in sorted(checkpoint.rglob("*")) if path.is_file()}


def implementation_hashes():
    paths = [Path(__file__), Path(__file__).with_name("train_act4_scratch.py")]
    paths += [PROJECT_ROOT / name for name in
              ("train_with_smoothness.py", "train_with_image_contract.py",
               "train_with_video_cache.py", "train_with_compiled_video_cache.py",
               "prepare_video_cache.py", "image_preprocessing.py")]
    return {path.relative_to(PROJECT_ROOT).as_posix(): file_hash(path) for path in paths}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_data_identity(config, source_audit, cache_path):
    """Bind numeric labels and the complete cache to the audited training source."""
    dataset = Path(config["dataset"]["root"]).resolve(strict=True)
    derivation = source_audit["regularization"]["action_label_derivation"]
    label_manifest = dataset / derivation["manifest"]
    require(label_manifest.resolve().parent == dataset
            and file_hash(label_manifest) == derivation["sha256"],
            "Use the source checkpoint's unchanged action-label derivation.")
    cache_path = Path(cache_path).resolve(strict=True)
    require(file_hash(cache_path) == source_audit["cache_manifest_sha256"],
            "Use the original verified ACT4 training cache manifest.")
    cache = read_json(cache_path)
    require(cache["schema_version"] == 1 and cache["status"] == "verified"
            and Path(cache["dataset_root"]).resolve() == dataset,
            "The training cache belongs to another dataset.")
    files = {}

    def include(root, relative, record):
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root), "A dataset/cache record escapes its declared root.")
        previous = files.get(path)
        require(previous is None or previous["sha256"] == record["sha256"],
                "Label and cache manifests disagree about their source video.")
        files[path] = record

    for relative, record in read_json(label_manifest)["output_files_before_manifest"].items():
        include(dataset, relative, record)
    for entry in cache["videos"]:
        include(dataset, entry["source_video"], entry["source"])
        include(cache_path.parent, entry["frames_file"], entry["frames"])
        include(cache_path.parent, entry["timestamps_file"], entry["timestamps"])
    require(any(path.suffix == ".parquet" for path in files), "Numeric training tables are missing.")
    for path, record in files.items():
        stat = path.stat()
        require(stat.st_size == record["size_bytes"] and file_hash(path) == record["sha256"],
                f"The audited dataset/cache file changed: {path}")
        require("mtime_ns" not in record or stat.st_mtime_ns == record["mtime_ns"],
                f"The original cache file fingerprint changed: {path}")
    return {"label_manifest_sha256": file_hash(label_manifest),
            "cache_manifest_sha256": file_hash(cache_path),
            "files_sha256": {str(path): record["sha256"] for path, record in sorted(files.items())},
            "full_hash_verified": True}


def prepare_plan(args):
    """Perform deterministic file checks before any policy factory is imported."""
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output_dir.resolve()
    require(args.checkpoint.is_absolute() and args.output_dir.is_absolute(),
            "Use absolute checkpoint and output paths.")
    require(not args.checkpoint.is_symlink() and checkpoint.name.isdigit(),
            "Use a numbered checkpoint directory, not the last symlink.")
    require(args.main_lr in (1e-4, 1e-5) and type(args.main_lr) is float,
            "Use main-lr 1e-4 or 1e-5 for this comparison.")
    require(output.parent == RUN_ROOT.resolve(),
            "Use a direct child branch directory of this ACT4 run, outside datasets/checkpoints.")
    require(args.resume_branch or (not output.is_relative_to(checkpoint)
                                  and not checkpoint.is_relative_to(output)),
            "A new branch output must be separate from its source.")
    state = read_json(checkpoint / "training_state/training_step.json")
    require(state.get("batch_size") == 32 and state.get("num_processes") == 1,
            "Preserve batch32 and one-process sample order.")
    require(type(state.get("step")) is int and state["step"] == int(checkpoint.name)
            and 7500 <= state["step"] < args.target_step,
            "Target total steps must exceed this complete checkpoint step.")
    model = checkpoint / "pretrained_model"
    required = {"pretrained_model/" + name for name in
                ("config.json", "train_config.json", "model.safetensors",
                 "policy_preprocessor.json", "policy_postprocessor.json",
                 "image_preprocessing.json", "random_initialization.json",
                 "training_regularization.json")}
    required.update("training_state/" + name for name in
                    ("training_step.json", "optimizer_param_groups.json",
                     "optimizer_state.safetensors", "rng_state.safetensors"))
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        for processor in read_json(model / name)["steps"]:
            if processor.get("state_file"):
                relative = PurePosixPath(processor["state_file"])
                require(not relative.is_absolute() and ".." not in relative.parts,
                        "Processor state must stay in the model directory.")
                required.add("pretrained_model/" + relative.as_posix())
    hashes = checkpoint_hashes(checkpoint)
    require(required <= hashes.keys(), "Checkpoint is missing a complete training/model asset.")
    audit = read_json(args.source_audit)
    require(audit.get("status") == "resume_preflight_passed" and audit.get("resume_step") == 7500,
            "Use the verified complete7500 source preflight.")
    for name in audit["required_checkpoint_sha256"]:
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts,
                "Source audit paths must stay in the checkpoint.")
    require(required <= audit["required_checkpoint_sha256"].keys(),
            "Source audit must identify every required asset.")
    split = args.split_manifest.resolve(strict=True)
    probe = args.compiled_probe.resolve(strict=True)
    require(file_hash(split) == audit["initialization"]["split_manifest_sha256"],
            "Use the unchanged ACT4 54/6 split.")
    require(file_hash(probe) == audit["compiled_probe_sha256"],
            "Use the already verified compiled batch32 probe.")
    source_config = read_json(model / "train_config.json")
    code_hashes = implementation_hashes()
    cache_path = os.environ.get("ACT_VIDEO_CACHE_MANIFEST", str(RUN_ROOT / "video_cache/manifest.json"))
    data_identity = verify_data_identity(source_config, audit, cache_path)
    groups = read_json(checkpoint / "training_state/optimizer_param_groups.json")
    require(len(groups) == 2 and [len(group["params"]) for group in groups] == [133, 20],
            "Retain the ACT main/backbone AdamW partition.")
    if args.resume_branch:
        require(checkpoint.parent == output / "checkpoints"
                and (checkpoint.parent / "last").resolve(strict=True) == checkpoint,
                "Resume the latest checkpoint inside this branch output.")
        require(not any(path.name.isdigit() and int(path.name) > state["step"]
                        for path in checkpoint.parent.iterdir()), "A higher branch checkpoint exists.")
        metadata = read_json(model / METADATA)
        saved_hashes = read_json(checkpoint / ASSET_MANIFEST)["files_sha256"]
        require(saved_hashes == {name: value for name, value in hashes.items()
                                 if name != ASSET_MANIFEST},
                "Branch checkpoint assets changed after complete publication.")
        require(metadata == read_json(output / METADATA), "Branch root/checkpoint metadata differ.")
        require(metadata["output_dir"] == str(output) and metadata["main_lr"] == args.main_lr
                and metadata["target_step"] == args.target_step
                and metadata["source_audit_sha256"] == file_hash(args.source_audit)
                and metadata["source_checkpoint_sha256"] == audit["required_checkpoint_sha256"]
                and metadata["data_identity"] == data_identity
                and metadata["implementation_sha256"] == code_hashes,
                "Resume must retain this branch's LR, target, and common source.")
        expected_lrs = [args.main_lr, BACKBONE_LR]
    else:
        require(not output.exists() and state["step"] == 7500,
                "Create a new empty output from the complete7500 source.")
        require(all(hashes.get(name) == digest
                    for name, digest in audit["required_checkpoint_sha256"].items()),
                "Source checkpoint differs from the audited complete7500 assets.")
        metadata = {"schema": "act4-lr-branch-v1", "source_step": 7500,
                    "source_checkpoint": str(checkpoint),
                    "source_checkpoint_sha256": audit["required_checkpoint_sha256"],
                    "source_audit_sha256": file_hash(args.source_audit),
                    "data_identity": data_identity,
                    "implementation_sha256": code_hashes,
                    "split_manifest_sha256": file_hash(split),
                    "compiled_probe_sha256": file_hash(probe),
                    "output_dir": str(output), "main_lr": args.main_lr,
                    "backbone_lr": BACKBONE_LR, "target_step": args.target_step,
                    "batch_size": 32, "world_size": 1, "eval_and_save_freq": 500,
                    "entry_sha256": file_hash(Path(__file__))}
        expected_lrs = [1e-4, BACKBONE_LR]
    require([group["lr"] for group in groups] == expected_lrs,
            "The checkpoint's saved optimizer LR differs from its branch/source.")
    return {"checkpoint": checkpoint, "output": output, "step": state["step"],
            "metadata": metadata, "hashes": hashes, "source_lrs": expected_lrs,
            "source_config": source_config,
            "split": split, "probe": probe}


def training_arguments(plan):
    metadata = plan["metadata"]
    return ["--split-manifest", str(plan["split"]), "--vision-init", "imagenet",
            "--smoothness-weight", "0.05", "--compile-batch-size", "32",
            "--config_path=" + str(plan["checkpoint"] / "pretrained_model/train_config.json"),
            "--resume=true", "--output_dir=" + str(plan["output"]),
            "--steps=" + str(metadata["target_step"]), "--eval_steps=500", "--save_freq=500",
            "--policy.optimizer_lr=" + str(metadata["main_lr"]),
            "--optimizer.lr=" + str(metadata["main_lr"])]


def validate_config(cfg, plan):
    require(cfg.resume and Path(cfg.checkpoint_path).resolve() == plan["checkpoint"]
            and Path(cfg.output_dir).resolve() == plan["output"]
            and Path(cfg.policy.pretrained_path).resolve() == plan["checkpoint"] / "pretrained_model",
            "The official resolver selected another checkpoint/output.")
    source = plan["source_config"]
    require(cfg.batch_size == 32 and cfg.seed == 42 and cfg.scheduler is None
            and cfg.policy.type == "act" and cfg.optimizer.type == "adamw" and not cfg.policy.use_amp
            and cfg.policy.chunk_size == cfg.policy.n_action_steps == 100
            and cfg.policy.kl_weight == 10 and cfg.policy.optimizer_lr_backbone == BACKBONE_LR,
            "Retain the source ACT4 batch, seed, constant AdamW, FP32, chunk, and KL.")
    require(cfg.policy.optimizer_lr == cfg.optimizer.lr == plan["metadata"]["main_lr"],
            "The parsed policy/optimizer LR must match this branch's requested main LR.")
    require(str(cfg.dataset.root) == source["dataset"]["root"]
            and cfg.dataset.repo_id == source["dataset"]["repo_id"]
            and cfg.dataset.use_imagenet_stats and not cfg.dataset.image_transforms.enable
            and cfg.num_workers == source["num_workers"]
            and cfg.steps == plan["metadata"]["target_step"]
            and cfg.eval_steps == cfg.save_freq == 500 and cfg.max_eval_samples == 0,
            "Retain the source dataset, normalization, loader and complete held-out evaluation.")
    require(not cfg.wandb.enable and not cfg.policy.push_to_hub and not cfg.save_checkpoint_to_hub,
            "Use independent local cloud outputs.")


def apply_restored_lr(optimizer, policy, cfg, plan, step):
    """Change deterministic scalar LR fields after the complete official restore."""
    require(step == plan["step"], "The restored training step differs from its source.")
    parameters = [(name, parameter) for name, parameter in policy.named_parameters()
                  if parameter.requires_grad]
    partitions = [[(name, parameter) for name, parameter in parameters
                   if name.startswith("model.backbone") == backbone] for backbone in (False, True)]
    groups = optimizer.param_groups
    require(len(groups) == 2 and [len(group["params"]) for group in groups] == [133, 20],
            "The restored optimizer has another parameter partition.")
    for group, partition, expected_lr in zip(groups, partitions, plan["source_lrs"]):
        require([id(parameter) for parameter in group["params"]]
                == [id(parameter) for _, parameter in partition]
                and len({id(parameter) for parameter in group["params"]}) == len(partition)
                and group["lr"] == expected_lr, "Restored main/backbone parameter identities or LR differ.")
    steps = [float(state["step"].item()) for state in optimizer.state.values()]
    require(len(steps) == 153 and all(value == step for value in steps),
            "Every AdamW parameter must retain its restored step.")
    main_lr = plan["metadata"]["main_lr"]
    groups[0]["lr"] = main_lr
    if "initial_lr" in groups[0]:
        groups[0]["initial_lr"] = main_lr
    optimizer.defaults["lr"] = main_lr
    cfg.policy.optimizer_lr = cfg.optimizer.lr = policy.config.optimizer_lr = main_lr
    require(policy.config.optimizer_lr_backbone == BACKBONE_LR,
            "The live policy's visual LR differs from the branch.")
    return {"restored_step": step, "source_group_lrs": plan["source_lrs"],
            "actual_group_lrs": [group["lr"] for group in groups],
            "parameter_names": [[name for name, _ in partition] for partition in partitions],
            "optimizer_step_count": len(steps), "optimizer_step_values": sorted(set(steps)),
            "scope": "Official complete restore followed by scalar main LR/config edits; no extra RNG draw or optimizer update."}


@contextmanager
def branch_hooks(trainer, scratch, plan, optimizer_type):
    """Compose with the existing wrappers and restore every alias on all exits."""
    originals = (scratch.make_explicit_datasets, trainer.make_policy,
                 trainer.load_training_state, trainer.save_checkpoint,
                 trainer.update_last_checkpoint)
    captured = {}

    def datasets(cfg, *args, **kwargs):
        validate_config(cfg, plan)
        captured["cfg"] = cfg
        return originals[0](cfg, *args, **kwargs)

    def policy_factory(*args, **kwargs):
        policy = originals[1](*args, **kwargs)
        captured["policy"] = policy
        return policy

    def restore(checkpoint_dir, optimizer, scheduler, load_optimizer=True):
        require(load_optimizer and scheduler is None and isinstance(optimizer, optimizer_type),
                "Require complete single-process AdamW restore with no scheduler/FSDP.")
        require(Path(checkpoint_dir).resolve() == plan["checkpoint"], "Unexpected restore source.")
        step, restored_optimizer, restored_scheduler = originals[2](
            checkpoint_dir, optimizer, scheduler, load_optimizer=load_optimizer)
        require(restored_optimizer is optimizer and restored_scheduler is None,
                "The official loader replaced the optimizer/scheduler unexpectedly.")
        audit = apply_restored_lr(optimizer, captured["policy"], captured["cfg"], plan, step)
        captured["restored"] = True
        atomic_json(plan["output"] / METADATA, plan["metadata"])
        atomic_json(plan["output"] / f"lr_restore_{step:06d}.json", audit)
        print("ACT4_LR_BRANCH_RESTORED=" + json.dumps(audit), flush=True)
        return step, restored_optimizer, restored_scheduler

    def save(*args, **kwargs):
        require(captured.get("restored"), "A branch checkpoint requires the completed restore hook.")
        checkpoint = Path(kwargs["checkpoint_dir"]).resolve()
        require(checkpoint.parent == plan["output"] / "checkpoints",
                "Branch checkpoints must stay inside the independent output.")
        require([group["lr"] for group in kwargs["optimizer"].param_groups]
                == [plan["metadata"]["main_lr"], BACKBONE_LR], "Live group learning rates changed.")
        result = originals[3](*args, **kwargs)
        atomic_json(checkpoint / "pretrained_model" / METADATA, plan["metadata"])
        return result

    def publish(checkpoint_dir, *args, **kwargs):
        # All outer ACT4 wrappers have now finished writing their metadata.
        checkpoint = Path(checkpoint_dir).resolve()
        require(checkpoint.parent == plan["output"] / "checkpoints",
                "Publish only this branch's checkpoint.")
        hashes = checkpoint_hashes(checkpoint)
        hashes.pop(ASSET_MANIFEST, None)
        atomic_json(checkpoint / ASSET_MANIFEST, {"files_sha256": hashes})
        return originals[4](checkpoint_dir, *args, **kwargs)

    scratch.make_explicit_datasets = datasets
    trainer.make_policy, trainer.load_training_state, trainer.save_checkpoint = policy_factory, restore, save
    trainer.update_last_checkpoint = publish
    try:
        yield
        require(captured.get("restored"), "Training returned without restoring the branch state.")
    finally:
        (scratch.make_explicit_datasets, trainer.make_policy,
         trainer.load_training_state, trainer.save_checkpoint,
         trainer.update_last_checkpoint) = originals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--main-lr", type=float, choices=(1e-4, 1e-5), required=True)
    parser.add_argument("--target-step", type=int, default=12500)
    parser.add_argument("--source-audit", type=Path, default=RUN_ROOT / "resume_preflight_007500.json")
    parser.add_argument("--split-manifest", type=Path, default=RUN_ROOT / "split_manifest.json")
    parser.add_argument("--compiled-probe", type=Path,
                        default=RUN_ROOT / "probe/benchmark_imagenet_compiled.json")
    parser.add_argument("--resume-branch", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    plan = prepare_plan(args)
    if args.check_only:
        print(json.dumps({"status": "branch_files_verified", "model_loads": 0,
                          "optimizer_updates": 0, "metadata": plan["metadata"],
                          "training_arguments": training_arguments(plan)}, indent=2))
        return
    require(sys.platform == "linux" and os.environ.get("WORLD_SIZE", "1") == "1",
            "Actual training requires one AutoDL Linux process.")
    import fcntl
    import torch
    require(importlib.metadata.version("lerobot") == "0.6.1"
            and torch.__version__ == "2.10.0+cu128" and torch.cuda.is_available()
            and torch.cuda.device_count() == 1, "Use the pinned single-GPU cloud environment.")
    os.environ.setdefault("ACT_VIDEO_CACHE_MANIFEST", str(RUN_ROOT / "video_cache/manifest.json"))
    import train_act4_scratch as scratch
    from lerobot.scripts import lerobot_train
    from prepare_video_cache import validate_cache_manifest
    cache = validate_cache_manifest(os.environ["ACT_VIDEO_CACHE_MANIFEST"], full_hash=False)
    require(cache["status"] == "verified"
            and Path(cache["dataset_root"]).resolve() == Path(plan["source_config"]["dataset"]["root"]).resolve(),
            "Use the existing verified cache for the source smoothed-label dataset.")
    with (RUN_ROOT / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Recheck after taking the shared training owner lock.
        plan = prepare_plan(args)
        if not args.resume_branch:
            plan["output"].mkdir(parents=True, exist_ok=False)
        previous_argv = sys.argv
        try:
            sys.argv = [str(Path(__file__)), *training_arguments(plan)]
            with branch_hooks(lerobot_train, scratch, plan, torch.optim.AdamW):
                scratch.main()
        finally:
            sys.argv = previous_argv
            require(checkpoint_hashes(plan["checkpoint"]) == plan["hashes"],
                    "The read-only source checkpoint changed during branch execution.")


if __name__ == "__main__":
    main()
