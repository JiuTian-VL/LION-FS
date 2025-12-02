import submitit, functools, transformers
from dataclasses import asdict, dataclass
from models.vision_live import build_live_vision

from models.configuration_live import LiveConfigMixin
from models.arguments_live import LiveOnePlusTrainingArguments
from ..utils import distributed_encode

import os
# print(os.getenv('CUDA_VISIBLE_DEVICES'))


@dataclass
class LiveOnePlusEncodingArguments(LiveOnePlusTrainingArguments):
    num_nodes: int = 1
    num_gpus: int = 1
    video_dir: str = 'egoexo_data_384ss/takes_2fps_max384' 
    # video_dir: str = 'egoexo_data_448ss/take'
    slurm_partition: str = None

if __name__ == "__main__":
    args, = transformers.HfArgumentParser(LiveOnePlusEncodingArguments).parse_args_into_dataclasses()
    vision_config = LiveConfigMixin(**asdict(args))
    _, vision_encode,_ = build_live_vision(vision_config)
    task = functools.partial(
        distributed_encode, src_root=args.video_dir, 
        vision_pretrained=args.vision_pretrained, 
        embed_mark=args.embed_mark, 
        vision_encode=vision_encode, 
        batch_size=256, save_bf16=True
    )
    executor = submitit.AutoExecutor(folder=f"outputs/preprocess/")
    # executor.update_parameters(
    #     tasks_per_node=args.num_gpus,
    #     nodes=args.num_nodes,
    #     gpus_per_node=args.num_gpus,
    #     cpus_per_task=10,
    #     slurm_partition=args.slurm_partition,
    #     mem_gb=240,
    #     slurm_time='24:00:00',
    # )
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

