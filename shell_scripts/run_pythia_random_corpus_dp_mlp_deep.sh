#!/bin/bash -l

#SBATCH -n 4
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --gpus=rtx_3090:1

module load eth_proxy
module load gcc/9.3.0
module load cuda/12.1.1
conda activate probe
wandb login

export TASK_NAME=ner
export CUDA_LAUNCH_BLOCKING=1

python3 ../run_pythia_dp.py \
  --seed 42 \
  --n_gpu 1 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --data_dir ../dataset/ontonotes/dp/ \
  --task $TASK_NAME \
  --output_dir ../outputs/pythia/mlp/$TASK_NAME/ \
  --overwrite_output_dir \
  --cache_dir ../cache/ \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --num_train_epochs 20.0 \
  --learning_rate 1e-5 \
  --mlp_dim 512 \
  --mlp_layers 4 \
  --use_mlp True \
  --dev \
  --scale 160m \
  --revision main \
  --model_path ../ckpt/pythia/step0/pythia-epoch-1/ \

