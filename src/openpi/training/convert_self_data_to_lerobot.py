"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.
Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""
import tarfile
import dataclasses
from pathlib import Path
import shutil
from typing import Literal
import json
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
import os
import glob
import cv2
import shutil 
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    save_path: Path,    
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]
    cameras = [
        "cam_high",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }


    return LeRobotDataset.create(
        repo_id=repo_id,
        root= save_path,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

def load_raw_episode_data(
    ep_path: Path
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    

 # 使用新的解压目录作为 base path
    # state_arm_right
    state_path = ep_path/"joint"
    json_files = sorted([f for f in os.listdir(state_path) if f.endswith('.json')])
    positions = []
    for json_file in json_files:
        file_path = os.path.join(state_path , json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            if 'position' in data:  
                positions.append(data['position'])
    state = np.array(positions, dtype=np.float32)

    # front cammera 
    front_camera_path = ep_path/"color/head"
    image_files = sorted([
    f for f in glob.glob(os.path.join(front_camera_path, "*"))
    if f.lower().endswith((".png", ".jpg"))])

    front_camera_images = []
    for image_file in image_files:
        image_path = os.path.join(front_camera_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        front_camera_images.append(image)

    # left cammera 
    left_camera_path =  ep_path/"color/hand"
    image_files = sorted([
    f for f in glob.glob(os.path.join(left_camera_path, "*"))
    if f.lower().endswith((".png", ".jpg"))])

    left_camera_images = []
    for image_file in image_files:
        image_path = os.path.join(left_camera_path, image_file)
        image = cv2.imread(image_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        left_camera_images.append(image)

    # action
    state_len = len(state)
    action = state.copy()[1:state_len]
    state = state[:state_len-1]


    #  camera
    min_length = min(len(front_camera_images), len(left_camera_images), len(state), len(positions)) 
    imgs_per_cam = {
    "cam_high": front_camera_images[:min_length], 
    "cam_right_wrist": left_camera_images[:min_length],  
    }

    state = state[:min_length]
    action = action[:min_length]

    velocity = None
    effort = None

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    episode_dirs: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(episode_dirs))

    for ep_idx in tqdm.tqdm(episodes):

        ep_path = episode_dirs[ep_idx]
        print(ep_path)

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        for i in range(num_frames):

            if i == len(state) - 1:
                continue
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }
            for camera, img_array in imgs_per_cam.items():

                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
            if task is not None:
                frame["task"] = task

            dataset.add_frame(frame)

        dataset.save_episode()
    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    save_path: Path,
    raw_repo_id: str | None = None,
    task: str = "prompt",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):

    episode_files = sorted([f for f in raw_dir.glob("ep*")])
    print(f"Found {len(episode_files)} episode directories.")

    dataset = create_empty_dataset(
        repo_id,
        save_path=save_path,
        robot_type="piper",
        mode=mode,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        episode_files,
        task='Grasp the pink dish from the base',  # 这里需要修改成实际的任务prompt
        episodes=episodes,
    )
    print("populating dataset done")


#python convert_piper_data_to_lerobot.py --repo-id 0
if __name__ == "__main__":
    raw_dir = Path("/path/to//raw_data/")
    repo_id = "data_id"
    save_dir = Path("/path/to/save_dir/")
    save_path= save_dir / repo_id
    tyro.cli(port_aloha(raw_dir = raw_dir, repo_id = repo_id, save_path = save_path))
