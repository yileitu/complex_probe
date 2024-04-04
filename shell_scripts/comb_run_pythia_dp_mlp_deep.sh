#!/bin/bash

# 定义参数的取值范围
declare -a revisions=("step0" "step128")
declare -a mlp_layers_values=(2 4 8 16 32)
declare -a learning_rates=(1e-4 5e-5 1e-5 5e-6)

# 遍历所有可能的参数组合
for revision in "${revisions[@]}"
do
  for mlp_layer in "${mlp_layers_values[@]}"
  do
    for lr in "${learning_rates[@]}"
    do
      # 生成SBATCH脚本
      sbatch <<EOT
#!/bin/bash -l

#SBATCH -n 4
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --gpus=rtx_2080_ti:1

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
  --fp16 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --data_dir ../dataset/ontonotes/dp/ \
  --task \$TASK_NAME \
  --output_dir ../outputs/pythia/mlp/\$TASK_NAME/ \
  --overwrite_output_dir \
  --cache_dir ../cache/ \
  --save_strategy no \
  --evaluation_strategy epoch \
  --num_train_epochs 150.0 \
  --learning_rate $lr \
  --mlp_dim 512 \
  --mlp_layers $mlp_layer \
  --use_mlp True \
  --dev \
  --scale 160m \
  --revision $revision
EOT
    done
  done
done
