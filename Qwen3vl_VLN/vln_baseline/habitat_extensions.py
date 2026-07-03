import gzip
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
from fastdtw import fastdtw
from habitat.config.default_structured_configs import DatasetConfig, MeasurementConfig
from habitat.core.dataset import Dataset
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import DistanceToGoal, NavigationGoal, Success
from habitat.tasks.vln.vln import InstructionData, VLNEpisode
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import attr


DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
ALL_LANGUAGES_MASK = "*"
ALL_ROLES_MASK = "*"


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


@registry.register_measure
class NDTW(Measure):
    cls_uuid = "ndtw"

    def __init__(self, *args, sim: Simulator, config, **kwargs):
        self._sim = sim
        self._success_distance = config.success_distance
        dataset = kwargs["dataset"]
        split = dataset.config["split"]
        if "{role}" in config.gt_path:
            self.gt_json = {}
            for role in getattr(dataset, "annotation_roles", ["guide"]):
                path = config.gt_path.format(split=split, role=role)
                if os.path.exists(path):
                    with gzip.open(path, "rt") as f:
                        self.gt_json.update(json.load(f))
        else:
            with gzip.open(config.gt_path.format(split=split), "rt") as f:
                self.gt_json = json.load(f)
        super().__init__()

    def _get_uuid(self, *args, **kwargs) -> str:
        return self.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.locations = []
        self.gt_locations = self.gt_json[str(episode.episode_id)]["locations"]
        self.update_metric()

    def update_metric(self, *args, **kwargs):
        current_position = self._sim.get_agent_state().position.tolist()
        if not self.locations or current_position != self.locations[-1]:
            self.locations.append(current_position)
        dtw_distance = fastdtw(self.locations, self.gt_locations, dist=euclidean_distance)[0]
        self._metric = float(np.exp(-dtw_distance / (len(self.gt_locations) * self._success_distance)))


@registry.register_measure
class SDTW(Measure):
    cls_uuid = "sdtw"

    def _get_uuid(self, *args, **kwargs) -> str:
        return self.cls_uuid

    def reset_metric(self, *args, task: EmbodiedTask, **kwargs):
        task.measurements.check_measure_dependencies(self.uuid, [NDTW.cls_uuid, Success.cls_uuid])
        self.update_metric(task=task)

    def update_metric(self, *args, task: EmbodiedTask, **kwargs):
        success = task.measurements.measures[Success.cls_uuid].get_metric()
        ndtw = task.measurements.measures[NDTW.cls_uuid].get_metric()
        self._metric = float(success) * float(ndtw)


@attr.s(auto_attribs=True)
class ExtendedInstructionData:
    instruction_text: str = attr.ib(default=None)
    instruction_id: Optional[str] = attr.ib(default=None)
    language: Optional[str] = attr.ib(default=None)
    annotator_id: Optional[str] = attr.ib(default=None)
    edit_distance: Optional[float] = attr.ib(default=None)
    timed_instruction: Optional[List[dict]] = attr.ib(default=None)
    instruction_tokens: Optional[List[str]] = attr.ib(default=None)
    split: Optional[str] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class VLNExtendedEpisode(VLNEpisode):
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    reference_path: Optional[List[List[float]]] = attr.ib(default=None)
    instruction: ExtendedInstructionData = attr.ib(default=None)
    trajectory_id: Optional[Union[int, str]] = attr.ib(default=None)


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


@registry.register_dataset(name="RxRVLNCE-v1")
class RxRVLNCEDataset(Dataset):
    annotation_roles = ["guide", "follower"]
    languages = ["en-US", "en-IN", "hi-IN", "te-IN"]

    def __init__(self, config: Optional[DictConfig] = None) -> None:
        self.config = config
        self.episodes = []
        if config is None:
            return
        self.annotation_roles = self.extract_roles_from_config(config)
        for role in self.annotation_roles:
            with gzip.open(config.data_path.format(split=config.split, role=role), "rt") as f:
                self.from_json(f.read(), scenes_dir=config.scenes_dir)
        if ALL_LANGUAGES_MASK not in config.languages:
            languages = set(config.languages)
            self.episodes = [ep for ep in self.episodes if ep.instruction.language in languages]

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        deserialized = json.loads(json_str)
        for episode_data in deserialized["episodes"]:
            episode = VLNExtendedEpisode(**episode_data)
            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[len(DEFAULT_SCENE_PATH_PREFIX) :]
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
            episode.instruction = ExtendedInstructionData(**episode.instruction)
            episode.instruction.split = self.config.split
            if episode.goals is not None:
                episode.goals = [NavigationGoal(**goal) for goal in episode.goals]
            self.episodes.append(episode)

    @classmethod
    def extract_roles_from_config(cls, config: DictConfig) -> List[str]:
        if ALL_ROLES_MASK in config.roles:
            return cls.annotation_roles
        return list(config.roles)


cs = ConfigStore.instance()


@dataclass
class OracleSuccessMeasurementConfig(MeasurementConfig):
    type: str = "OracleSuccess"
    success_distance: float = 3.0


@dataclass
class NDTWMeasurementConfig(MeasurementConfig):
    type: str = "NDTW"
    success_distance: float = 3.0
    gt_path: str = ""


@dataclass
class SDTWMeasurementConfig(MeasurementConfig):
    type: str = "SDTW"


@dataclass
class R2RVLNCEDatasetConfig(DatasetConfig):
    type: str = "R2RVLNCE-v1"
    split: str = "val_unseen"
    scenes_dir: str = ""
    data_path: str = ""


@dataclass
class RxRVLNCEDatasetConfig(DatasetConfig):
    type: str = "RxRVLNCE-v1"
    split: str = "val_unseen"
    scenes_dir: str = ""
    roles: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    data_path: str = ""


cs.store(group="habitat/task/measurements", name="oracle_success", node=OracleSuccessMeasurementConfig)
cs.store(group="habitat/task/measurements", name="ndtw", node=NDTWMeasurementConfig)
cs.store(group="habitat/task/measurements", name="sdtw", node=SDTWMeasurementConfig)
cs.store(group="habitat/dataset", name="r2rvlnce_v1", node=R2RVLNCEDatasetConfig)
cs.store(group="habitat/dataset", name="rxrvlnce_v1", node=RxRVLNCEDatasetConfig)
