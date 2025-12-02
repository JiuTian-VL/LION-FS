from functools import partial
import submitit, transformers
from dataclasses import dataclass

from models.arguments_live import LiveOnePlusTrainingArguments
from ..utils import distributed_ffmpeg

import os
def find_target_files(video_dir, target_filename="aria01_214-1.mp4"):
    target_files = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file == target_filename:
                target_files.append(os.path.join(root, file))
    return target_files



@dataclass
class LiveOnePlusEncodingArguments(LiveOnePlusTrainingArguments):
    num_nodes: int = 1
    num_gpus: int = 2
    video_dir: str = 'egoexo_data_448ss/takes'
    slurm_partition: str = None
    


if __name__ == "__main__":
    args, = transformers.HfArgumentParser(LiveOnePlusEncodingArguments).parse_args_into_dataclasses()
    # target_files = find_target_files(args.video_dir)

    executor = submitit.AutoExecutor(folder=f"outputs/preprocess/")

    task = partial(distributed_ffmpeg, src_root=args.video_dir, resolution=args.frame_resolution, fps=args.frame_fps)
    executor.update_parameters(
        tasks_per_node=args.num_gpus,
        nodes=args.num_nodes,
        gpus_per_node=args.num_gpus,
        slurm_partition=args.slurm_partition,
        cpus_per_task=10,
        mem_gb=240,
        timeout_min=1440,  # 24 hours in minutes
    )
    job = executor.submit(task)

