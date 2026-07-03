import json

from scripts.extract_scalevln_trajectory_data import (
    episode_outputs_complete,
    make_annotation_subset,
    output_dir_name,
    select_annotations,
)


def test_output_dir_name_matches_streamvln_layout():
    annotation = {"video": "images/00000-kfPV7w3FaU5_scalevln_000000"}

    assert output_dir_name(annotation) == "00000-kfPV7w3FaU5_scalevln_000000"


def test_select_annotations_preserves_requested_ids_and_limit():
    annotations = [{"id": 0}, {"id": 51}, {"id": 102}, {"id": 153}]

    selected = select_annotations(annotations, episode_ids=[102, 0], limit=1)

    assert [item["id"] for item in selected] == [102]


def test_episode_outputs_complete_requires_exact_rgb_depth_pose_sets(tmp_path):
    episode_dir = tmp_path / "episode"
    for subdir, suffix in (("rgb", "jpg"), ("depth", "png"), ("pose", "json")):
        (episode_dir / subdir).mkdir(parents=True)
        for idx in range(1, 3):
            path = episode_dir / subdir / f"{idx:03d}.{suffix}"
            if suffix == "json":
                path.write_text("{}\n")
            else:
                path.write_bytes(b"data")

    assert episode_outputs_complete(episode_dir, frame_count=2)

    (episode_dir / "rgb" / "003.jpg").write_bytes(b"extra")

    assert not episode_outputs_complete(episode_dir, frame_count=2)


def test_make_annotation_subset_writes_selected_streamvln_annotations(tmp_path):
    output_path = tmp_path / "annotations_10k.json"
    annotations = [
        {
            "id": 0,
            "video": "images/00000-kfPV7w3FaU5_scalevln_000000",
            "instructions": ["go"],
            "actions": [-1, 1],
        },
        {
            "id": 51,
            "video": "images/00000-kfPV7w3FaU5_scalevln_000051",
            "instructions": ["turn"],
            "actions": [-1, 3],
        },
    ]

    make_annotation_subset(annotations, output_path)

    with output_path.open() as f:
        written = json.load(f)
    assert written == annotations
