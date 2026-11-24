"""ACT training with cached video and batch8-only compiled model forward.

The original model module, parameters, optimizer setup, saved state_dict and
image-contract trainer remain in place. Validation and incomplete batches use
the original eager forward.
"""

import sys

# Keep the existing import-time hook for spawned DataLoader workers.
from train_with_video_cache import main as video_cache_main


def compile_training_forward(policy):
    """Install dispatch on the same ACT module and return its eager forward."""
    import torch

    if policy.config.type != "act":
        raise ValueError("The compiled training dispatcher requires an ACT policy.")
    model = policy.model
    if not torch.cuda.is_available() or next(model.parameters()).device.type != "cuda":
        raise RuntimeError("The compiled ACT dispatcher requires a CUDA-resident cloud model.")
    original_keys = tuple(policy.state_dict())
    eager_forward = model.forward
    compiled_forward = torch.compile(eager_forward, mode="reduce-overhead", fullgraph=True)

    def dispatch(batch):
        if model.training and batch["observation.state"].shape[0] == 8:
            torch.compiler.cudagraph_mark_step_begin()
            return compiled_forward(batch)
        return eager_forward(batch)

    model.forward = dispatch
    if policy.model is not model or tuple(policy.state_dict()) != original_keys:
        raise RuntimeError("The compiled dispatcher changed model identity or state_dict keys.")
    return eager_forward


def main():
    if sys.platform != "linux":
        raise RuntimeError("Compiled ACT training is restricted to the Linux CUDA cloud runtime.")
    from lerobot.scripts import lerobot_train

    original_make_policy = lerobot_train.make_policy

    def make_policy(*args, **kwargs):
        policy = original_make_policy(*args, **kwargs)
        if policy.config.type == "act":
            compile_training_forward(policy)
            print("ACT_COMPILED_FORWARD=training_batch8; eval_and_tail=eager; state_dict_keys=unchanged",
                  flush=True)
        return policy

    lerobot_train.make_policy = make_policy
    try:
        video_cache_main()
    finally:
        lerobot_train.make_policy = original_make_policy


if __name__ == "__main__":
    main()
