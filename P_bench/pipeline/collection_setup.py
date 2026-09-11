"""Collection argument and public-setting setup shared by all backends."""

from __future__ import annotations

import json

from pipeline import action_proposal, config, pose_setting
from pipeline.collection_cli import validate_formal_fov_profiles
from pipeline.pose_calibration import load_pose_exclusions


def resolve_collection_mode_defaults(args, _parser) -> None:
    """Resolve policies once so run metadata and records cannot drift."""
    if args.lengths is None:
        args.lengths = list(config.GEN_LENGTHS)
    args.setting_sampling_policy = (
        pose_setting.SETTING_SAMPLING_POLICY
        if args.collection_mode == "main" else None)
    args.action_sampling_policy = action_proposal.policy_for_new_collection(
        getattr(args, "action_mode", None))


def active_radii_for_setting(setting, setting_policy: str | None) -> list[float]:
    """Resolve the body axis without conflating it with sensor sampling."""
    if setting is None:
        return [float(value) for value in config.RADII_M]
    return [float(setting.radius_m)]


def validate_collection_args(args, parser):
    """Validate the one supported collection contract and load private inputs."""
    resolve_collection_mode_defaults(args, parser)
    if list(args.lengths) != list(config.GEN_LENGTHS):
        parser.error("formal collection requires --lengths 1 2 3 4 5 6 exactly")
    if args.keep_per_length < 1:
        parser.error("--keep-per-length must be positive")
    if args.pose_candidates_per_scene <= 0:
        parser.error("pose candidate budget must be positive")
    if (args.collection_shard_id and
            not all(character.isalnum() or character in "-_"
                    for character in args.collection_shard_id)):
        parser.error(
            "--collection-shard-id must contain only letters, digits, - or _")
    try:
        pose_exclusions = load_pose_exclusions(args.pose_exclusions)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    try:
        fovs = validate_formal_fov_profiles(
            args.fov or config.BENCH_FOVS_DEG)
    except ValueError as error:
        parser.error(str(error))
    heights = [float(value) for value in args.camera_heights]
    return pose_exclusions, fovs, heights
