"""R2R-CE dataset registration and navigation metrics for Habitat."""

import gzip
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from habitat.config.default_structured_configs import DatasetConfig, MeasurementConfig
from habitat.core.dataset import Dataset
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import DistanceToGoal, NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


def euclidean_distance(pos_a, pos_b) -> float:
    return float(np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2))


@registry.register_measure
class PathLength(Measure):
    cls_uuid = "path_length"

    def __init__(self, sim: Simulator, *args, **kwargs):
        self._sim = sim
        super().__init__(**kwargs)

    def _get_uuid(self, *args, **kwargs) -> str:
        return self.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self._previous_position = self._sim.get_agent_state().position
        self._metric = 0.0

    def update_metric(self, *args, **kwargs):
        current_position = self._sim.get_agent_state().position
        self._metric += euclidean_distance(current_position, self._previous_position)
        self._previous_position = current_position


@registry.register_measure
class OracleNavigationError(Measure):
    cls_uuid = "oracle_navigation_error"

    def _get_uuid(self, *args, **kwargs) -> str:
        return self.cls_uuid

    def reset_metric(self, *args, task: EmbodiedTask, **kwargs):
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
        self._metric = float("inf")
        self.update_metric(task=task)

    def update_metric(self, *args, task: EmbodiedTask, **kwargs):
        distance = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = min(self._metric, distance)


@registry.register_measure
class OracleSuccess(Measure):
    cls_uuid = "oracle_success"

    def __init__(self, *args, config, **kwargs):
        self._success_distance = config.success_distance
        super().__init__()

    def _get_uuid(self, *args, **kwargs) -> str:
        return self.cls_uuid

    def reset_metric(self, *args, task: EmbodiedTask, **kwargs):
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
        self._metric = 0.0
        self.update_metric(task=task)

    def update_metric(self, *args, task: EmbodiedTask, **kwargs):
        distance = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = float(bool(self._metric) or distance < self._success_distance)


@registry.register_dataset(name="R2RVLNCE-v1")
class R2RVLNCEDataset(Dataset):
    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    def __init__(self, config: Optional[DictConfig] = None) -> None:
        self.config = config
        self.episodes = []
        if config is None:
            return
        with gzip.open(config.data_path.format(split=config.split), "rt") as f:
            self.from_json(f.read(), scenes_dir=config.scenes_dir)
        self.episodes = list(filter(self.build_content_scenes_filter(config), self.episodes))

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        deserialized = json.loads(json_str)
        self.instruction_vocab = VocabDict(word_list=deserialized["instruction_vocab"]["word_list"])
        for episode_data in deserialized["episodes"]:
            episode = VLNEpisode(**episode_data)
            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[len(DEFAULT_SCENE_PATH_PREFIX) :]
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
            episode.instruction = InstructionData(**episode.instruction)
            episode.goals = [NavigationGoal(**goal) for goal in episode.goals]
            self.episodes.append(episode)


@dataclass
class OracleSuccessMeasurementConfig(MeasurementConfig):
    type: str = "OracleSuccess"
    success_distance: float = 3.0


@dataclass
class R2RVLNCEDatasetConfig(DatasetConfig):
    type: str = "R2RVLNCE-v1"
    split: str = "val_unseen"
    scenes_dir: str = ""
    data_path: str = ""


cs = ConfigStore.instance()
cs.store(group="habitat/task/measurements", name="oracle_success", node=OracleSuccessMeasurementConfig)
cs.store(group="habitat/dataset", name="r2rvlnce_v1", node=R2RVLNCEDatasetConfig)
