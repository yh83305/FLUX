#!/usr/bin/env python3
"""Run the official FLUX SocialNav evaluator with interaction gate control."""

from __future__ import annotations

import argparse
import builtins
import json
import runpy
import sys
from pathlib import Path


def option_value(arguments: list[str], name: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        add_help=False,
        description="Interaction wrapper around eval_socialnav_wheeled.py",
    )
    parser.add_argument("--interaction_manifest", required=True)
    wrapper_args, official_args = parser.parse_known_args()

    manifest_path = Path(wrapper_args.interaction_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = option_value(official_args, "--output_dir")
    scene_dir = option_value(official_args, "--scene_dir")
    episode_start = int(option_value(official_args, "--episode_start") or 0)
    num_episodes = int(option_value(official_args, "--num_episodes") or 100)
    if output_dir is None:
        raise ValueError("--output_dir is required for interaction evaluation")
    if scene_dir is None:
        raise ValueError("--scene_dir is required for interaction evaluation")
    if Path(scene_dir).resolve() != Path(manifest["overlay_scene_dir"]).resolve().parent:
        raise ValueError(
            "--scene_dir must be the isaacsim_scene directory recorded by the interaction manifest"
        )
    if episode_start + num_episodes > int(manifest["num_episodes"]):
        raise ValueError("requested episode range exceeds the interaction manifest")

    from interaction_metrics import install_tracker_patch

    official_evaluator = Path(__file__).resolve().with_name("eval_socialnav_wheeled.py")
    if not official_evaluator.exists():
        raise FileNotFoundError(official_evaluator)
    print(
        f"[INTERACTION] manifest={manifest_path} episodes=[{episode_start}, "
        f"{episode_start + num_episodes})",
        flush=True,
    )
    sys.argv = [str(official_evaluator), *official_args]
    original_import = builtins.__import__
    patch_installed = False

    def interaction_import(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal patch_installed
        module = original_import(name, globals, locals, fromlist, level)
        if name == "socialnav_metrics" and not patch_installed:
            install_tracker_patch(
                sys.modules["socialnav_metrics"],
                str(manifest_path),
                output_dir,
                episode_start,
            )
            patch_installed = True
            print("[INTERACTION] runtime tracker patch installed", flush=True)
        return module

    builtins.__import__ = interaction_import
    try:
        runpy.run_path(str(official_evaluator), run_name="__main__")
    finally:
        builtins.__import__ = original_import


if __name__ == "__main__":
    main()
