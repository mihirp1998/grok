#!/usr/bin/env bash
#SBATCH --partition=kate_reserved
#SBATCH --job-name=gk
#SBATCH --output=/home/mprabhud/sp/digen_sp/logs/%A.out
#SBATCH --error=/home/mprabhud/sp/digen_sp/logs/%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --exclude=matrix-0-18,matrix-0-16,matrix-0-22,matrix-0-36
source /home/mprabhud/.bashrc
conda activate vlr
cd /home/mprabhud/sp/grok/
export PYTHONUNBUFFERED=1

# python run.py data=imagenet model=mlp batch_size=80 learning_rate=2e-2 mode=forward_forward
python main.py max_context_len=100 math_operator=x**3 n_heads=16 n_layers=12