#!/bin/bash

#SBATCH --job-name=ours_ver1_DomainNet
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH -p batch_agi
#SBATCH -w agi1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH --time=5-0
#SBATCH -o log/%N_%x_%j.out
#SBTACH -e %N_%x_%j.err 

export TORCH_DISTRIBUTED_DEBUG=INFO

num_nodes=2
epochs=5
master_port=29514

# domainnet_seed=(13185 19418 25862 29690 32443)
# core50_seed=(1024 1302 7081 19438 29665)
# pmnist_seed=(5722 20527 24728 26521 29031)
domainnet_seed=(29690)
core50_seed=(1024)
pmnist_seed=(20527)
# python -m torch.distributed.launch \
#             --master_port 29519 \
#             --nproc_per_node=4 \
#             --use_env main.py \
#             cdil_dualprompt \
#             --epochs 5 \
#             --model vit_base_patch16_224 \
#             --batch-size 24 \
#             --data-path /local_datasets/ \
#             --output_dir ./output \
#             --dataset iDigits \
#             --class_group_size 2 \
#             --seed 42 \
#             --learning_type VIL \
#             --num_classes 10 \
#             --num_tasks 20 \
#             --print_tasks \
#             --d_prompt \
#             --alpha 1.0 \
#             --eval_only_last \


echo "-------------------------------------------------CORe50-------------------------------------------------"
for seed in "${core50_seed[@]}"
do
    python -m torch.distributed.launch \
                    --master_port $master_port \
                    --nproc_per_node=$num_nodes\
                    --use_env main.py \
                    vil_ours \
                    --epochs $epochs \
                    --model vit_base_patch16_224 \
                    --batch-size 24 \
                    --data-path /local_datasets/ \
                    --output_dir ./output \
                    --dataset CORe50 \
                    --seed $seed \
                    --num_tasks 5 \
                    --versatile_inc \
                    --eval_only_last \
                    --d_prompt \
                    --size 40
done


echo "-------------------------------------------------Permuted_MNIST-------------------------------------------------"
for seed in "${pmnist_seed[@]}"
do
    python -m torch.distributed.launch \
            --master_port $master_port \
            --nproc_per_node=$num_nodes \
            --use_env main.py \
            vil_ours \
            --epochs $epochs \
            --model vit_base_patch16_224 \
            --batch-size 24 \
            --data-path /local_datasets/ \
            --output_dir ./output \
            --dataset PermutedMNIST \
            --seed $seed \
            --num_tasks 5 \
            --versatile_inc \
            --eval_only_last \
            --d_prompt \
            --size 25
  
done

echo "-------------------------------------------------DomainNET-------------------------------------------------"
for seed in "${domainnet_seed[@]}"
do
    python3 -m torch.distributed.launch \
                        --master_port $master_port \
                        --nproc_per_node=$num_nodes \
                        --use_env main.py \
                        vil_ours \
                        --epochs $epochs \
                        --model vit_base_patch16_224 \
                        --batch-size 24 \
                        --data-path /data/mypark/repo/cdil_original/temp \
                        --output_dir ./output \
                        --dataset DomainNet \
                        --seed $seed \
                        --eval_only_last \
                        --num_tasks 5 \
                        --size 30 \
                        --versatile_inc \
                        --d_prompt \
                        --d_prompt_
done