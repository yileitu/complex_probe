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

python3 train_pythia_from_scratch_on_random_corpus.py \
  --sentence_len 15 \
  --sentence_num 100000 \
  --model_name pythia \
  --scale 160m \
  --revision step0 \
  --n_gpu 1 \
  --fp16 \
  --cache_dir ../cache/ \
  --data_dir generated_corpus_len15_num100000.txt \
  --output_dir ckpt/ \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
