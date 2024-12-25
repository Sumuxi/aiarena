#!/bin/bash

#SBATCH --job-name=lw_exp1     # 任务名称
#SBATCH --output=output_%j.log     # 标准输出日志文件（%j 表示任务ID）
#SBATCH --error=error_%j.log       # 错误输出日志文件
#SBATCH --partition=IAI_SLURM_3090 # 分区名称
#SBATCH --nodelist=node006
#SBATCH --ntasks=1                 # 总任务数
#SBATCH --gres=gpu:2               # 指定GPU数量
#SBATCH --qos=8gpu                 # QOS
#SBATCH --cpus-per-task=100        # 每个任务使用的CPU核心数
#SBATCH --time 3-00:00:00          # 运行时间限制

source ~/.bashrc
conda activate lw
cd /nfs-shared-2/yxh/kd/lw_exp/aiarena/code

## 任务 1
#srun --nodelist=node003 --gres=gpu:1 --ntasks=1 --exclusive \
python learner/train.py \
      --local_rank=0 \
      --exp_name=v0e1 \
      --model_file=models.NetworkModel_v0 \
      --batch_size=256 \
      --temperature=4 \
      --lr_decay=0 \
      --lr_start=0.0002 > ../logs/v0e1_task${SLURM_JOB_ID}.log 2>&1 &

# 任务 2
#srun --gres=gpu:1 --ntasks=1 --exclusive \
#    CUDA_VISIBLE_DEVICES=1
python learner/train.py \
          --local_rank=1 \
          --exp_name=v5e1 \
          --model_file=models.arch5 \
          --batch_size=128 \
          --temperature=4 \
          --lr_decay=0 \
          --lr_start=0.0002 > ../logs/v5e1_task${SLURM_JOB_ID}.log 2>&1 &

# 等待所有任务完成
wait