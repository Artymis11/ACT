"""Run the existing ACT image-contract trainer with a verified video mmap cache."""

import os

from prepare_video_cache import install_video_cache, validate_cache_manifest


# Spawned DataLoader workers import this module as __mp_main__, so the hook must
# also be installed at module import time, before their dataset begins reading.
MANIFEST = os.environ.get("ACT_VIDEO_CACHE_MANIFEST")
if MANIFEST:
    install_video_cache(MANIFEST)


def main():
    if not MANIFEST:
        raise ValueError("Set ACT_VIDEO_CACHE_MANIFEST to the fully verified manifest.json.")
    manifest = validate_cache_manifest(MANIFEST, full_hash=True)
    print(f"ACT_VIDEO_CACHE_VERIFIED={MANIFEST} bytes={manifest['cache_bytes']}", flush=True)
    from train_with_image_contract import main as train_main
    train_main()


if __name__ == "__main__":
    main()
