#!/bin/bash

#SBATCH --job-name=ver1_dg&de_iDigits
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p batch
#SBATCH -w agi1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=30G
#SBATCH --time=7-0
#SBATCH -o log/%N_%x_%j.out
#SBTACH -e %N_%x_%j.err 

export TORCH_DISTRIBUTED_DEBUG=INFO


python -m torch.distributed.launch \
   --master_port 29509        \
   --nproc_per_node=1        \
   --use_env main.py    \
   vil_ours   \
   --epochs 5    \
   --model vit_base_patch16_224    \
   --batch-size 24      \
   --data-path /local_datasets/   \
   --output_dir ./output   \
   --dataset CORe50  \
   --seed 1024         \
   --d_prompt \
   --versatile_inc \
   --size 40




python -m torch.distributed.launch     \
   --nproc_per_node=1  \
   --master_port 12348  \
    --use_env main.py    \
    vil_ours  \
    --model vit_base_patch16_224  \
    --batch-size 24  \
    --data-path /local_datasets/   \
    --output_dir ./output \
    --dataset PermutedMNIST \
    --seed 20527 \
    --num_tasks 5 \
    --versatile_inc \
    --d_prompt


   python -m torch.distributed.launch     \
   --nproc_per_node=1  \
   --master_port 12348  \
    --use_env main.py    \
    cifar100_dualprompt  \
    --model vit_base_patch16_224  \
    --batch-size 24  \
    --data-path ./datasets/   \
    --output_dir ./output \
    --dataset DomainNet \
    --seed 29690 \
    --num_tasks 5 \
    --versatile_inc \
    --size 40


# python -m torch.distributed.launch \
#    --master_port 29509        \
#    --nproc_per_node=1        \
#    --use_env /data/pmy0792/VIL/ours/ver1/main.py    \
#    vil_ours   \
#    --epochs 5    \
#    --model vit_base_patch16_224    \
#    --batch-size 24      \
#    --data-path /local_datasets/   \
#    --output_dir ./output   \
#    --dataset iDigits  \
#    --seed 42         \
#    --d_prompt \
#    --versatile_inc \
#    --size 20