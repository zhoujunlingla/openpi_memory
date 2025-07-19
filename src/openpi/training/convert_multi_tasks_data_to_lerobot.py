"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.
Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""
import tarfile
import dataclasses
from pathlib import Path
import shutil
from typing import Literal, Union, Optional
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
    video_backend: Union[str, None] = None


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
        use_videos=False,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=5,
        image_writer_threads=5,
        video_backend=dataset_config.video_backend,
    )

def load_raw_episode_data(
    ep_path: Path
) -> tuple:
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

    # front camera (只返回图片路径)
    front_camera_path = ep_path/"color/head"
    front_image_files = sorted([
        f for f in glob.glob(os.path.join(front_camera_path, "*"))
        if f.lower().endswith((".png", ".jpg"))])

    # left camera (只返回图片路径)
    left_camera_path =  ep_path/"color/hand"
    left_image_files = sorted([
        f for f in glob.glob(os.path.join(left_camera_path, "*"))
        if f.lower().endswith((".png", ".jpg"))])

    # action
    state_len = len(state)
    action = state.copy()[1:state_len]
    state = state[:state_len-1]

    min_length = min(len(front_image_files), len(left_image_files), len(state), len(positions)) 
    front_image_files = front_image_files[:min_length]
    left_image_files = left_image_files[:min_length]
    state = state[:min_length]
    action = action[:min_length]

    velocity = None
    effort = None

    # 只返回图片路径列表
    return front_image_files, left_image_files, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    episode_paths_with_task: "list[tuple[Path, str]]", # 传入 (episode_path, task_name) 元组列表
) -> LeRobotDataset:
    import gc
    # 创建 / 读取进度文件，记录已经成功转换的 episode 路径，便于脚本因意外终止后可以自动跳过已处理的数据
    progress_file = Path(dataset.root) / "converted_episodes.txt"
    if progress_file.exists():
        # 优先使用进度文件逐条映射到原始 episode 路径，精确跳过
        with open(progress_file, "r", encoding="utf-8") as f:
            finished_eps = {line.strip() for line in f if line.strip()}
    else:
        finished_eps = set()

    # 如果进度文件不存在，则根据已经生成的数据文件数目来推断应当跳过的 episode 数量
    if not finished_eps:
        data_chunk_dir = Path(dataset.root) / "data" / "chunk-000"
        if data_chunk_dir.exists():
            parquet_files = sorted(data_chunk_dir.glob("episode_*.parquet"))
            if parquet_files:
                # 按文件顺序简单推断已完成数量
                skipped_num = len(parquet_files)
                print(f"[Resume] Detected {skipped_num} existing episode parquet files, will skip the first {skipped_num} raw episodes.")
            else:
                skipped_num = 0
        else:
            skipped_num = 0
    else:
        skipped_num = 0

    for idx, (ep_path, task_name) in enumerate(tqdm.tqdm(episode_paths_with_task, desc="Populating dataset with episodes")):
        # 如果有按文件数量推断的跳过策略
        if skipped_num and idx < skipped_num:
            continue

        # 如果进度文件里记录了原始路径，也跳过
        if str(ep_path) in finished_eps:
            continue
        print(f"Processing episode: {ep_path} for task: '{task_name}'")

        front_image_files, left_image_files, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        MAX_CHUNK_FRAMES = 50  # 每次最多缓存这么多帧，避免单个 episode 占用过大内存
        chunk_counter = 0

        for i in range(num_frames):
            if i == len(state) - 1:
                continue
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }
            # 逐帧读取图片
            if i < len(front_image_files):
                img_front = cv2.imread(front_image_files[i])
                img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB)
                frame["observation.images.cam_high"] = img_front
                del img_front
            if i < len(left_image_files):
                img_left = cv2.imread(left_image_files[i])
                img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
                frame["observation.images.cam_right_wrist"] = img_left
                del img_left

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
            frame["task"] = task_name

            dataset.add_frame(frame)

            chunk_counter += 1
            # 达到阈值就立刻写盘并清空内部缓冲
            if chunk_counter >= MAX_CHUNK_FRAMES:
                dataset.save_episode()
                try:
                    dataset.consolidate(run_compute_stats=False)
                except Exception as e:
                    print(f"[Warning] consolidate failed during chunk inside {ep_path}: {e}")
                chunk_counter = 0
 
        # 处理完本 raw episode 后，如果还有残余帧没落盘，再保存一次
        if chunk_counter:
            dataset.save_episode()
            try:
                dataset.consolidate(run_compute_stats=False)
            except Exception as e:
                print(f"[Warning] consolidate failed after episode {ep_path}: {e}")

        # 将已完成的 episode 路径追加到进度文件中
        try:
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(f"{ep_path}\n")
            finished_eps.add(str(ep_path))
        except Exception as e:
            print(f"[Warning] Failed to write progress file: {e}")
        # 在每个 episode 保存后执行 consolidate，及时写盘并释放内存，防止长时间运行占用过高内存
        try:
            dataset.consolidate(run_compute_stats=False)
        except Exception as e:
            # consolidate 可能在多线程/多进程环境中偶尔失败，为了不中断主流程，这里只打印警告
            print(f"[Warning] consolidate failed after episode {ep_path}: {e}")
        # 显式释放变量
        del front_image_files, left_image_files, state, action, velocity, effort
        gc.collect()
    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    save_path: Path,
    task_names_json: Path, # 新增参数：包含任务名称的 JSON 文件路径
    raw_repo_id: Union[str, None] = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    print(f"Scanning raw directory: {raw_dir}")
    
    # 1. 加载任务名称映射 JSON 文件
    if not task_names_json.exists():
        print(f"Error: Task names JSON file not found at {task_names_json}")
        return
    with open(task_names_json, 'r', encoding='utf-8') as f:
        task_name_map = json.load(f)
    print(f"Loaded task names from {task_names_json}: {task_name_map}")

    # 查找所有任务子目录 (例如 task1, task2)

    task_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith("ep")])
    print(f"Found {len(task_dirs)} task directories in {raw_dir}.")
    
    if not task_dirs:
        print(f"No task directories found in {raw_dir}. Exiting.")
        return

    all_episode_paths_with_task = []
    for task_dir in task_dirs:
        # 获取当前任务目录的名称，例如 "task1"
        task_key = task_dir.name
        # 从 JSON 映射中获取对应的任务描述
        actual_task_name = task_name_map.get(task_key, f"Unknown Task: {task_key}")
        
        # 在每个任务目录中查找所有 episode 子目录 (例如 ep1, ep2)
        episode_files_in_task = sorted([f for f in task_dir.glob("ep*") if f.is_dir()])
        
        if episode_files_in_task:
            print(f"Found {len(episode_files_in_task)} episodes in task '{task_key}' ({actual_task_name}).")
            for ep_path in episode_files_in_task:
                all_episode_paths_with_task.append((ep_path, actual_task_name))
        else:
            print(f"No 'ep*' directories found in task: {task_dir.name}. Skipping.")

    dataset = create_empty_dataset(
        repo_id,
        save_path=save_path,
        robot_type="piper",
        mode=mode,
        dataset_config=dataset_config,
    )
    

    dataset = populate_dataset(
        dataset,
        all_episode_paths_with_task, 
    )
    print("Populating dataset done.")
    print(f"Dataset saved to: {save_path}")



# main execution block
if __name__ == "__main__":
    # 假设你的目录结构如下:
    # ├── task1/
    # │   ├── ep1/
    # │   │   ├── joint/
    # │   │   └── color/
    # │   │       ├── head/
    # │   │       └── hand/
    # │   ├── ep2/
    # │   └── ...
    # ├── task2/
    # │   ├── ep1/
    # │   ├── ep2/
    # │   └── ...


    raw_dir_input = Path("/media/jarvan/F6EA463DEA45FA7F/demo")
    repo_id = "task1-16"  # 你可以根据需要修改这个 repo_id
    save_dir_base = Path("/media/jarvan/F6EA463DEA45FA7F/convert_data")
    save_path = save_dir_base / repo_id
    task_json_file = Path("/media/jarvan/F6EA463DEA45FA7F/demo/task_names.json") 


    tyro.cli(lambda: port_aloha(
        raw_dir=raw_dir_input, 
        repo_id=repo_id, 
        save_path=save_path,
        task_names_json=task_json_file
    ))