#!/bin/bash -l
#SBATCH --job-name=run_tinyimagenet_dsdsnn
#SBATCH --output=run_tinyimagenet_dsdsnn_%j.out
#SBATCH --error=run_tinyimagenet_dsdsnn_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -e
date

echo "加载用户环境..."
source ~/.bashrc

echo "加载conda环境..."
module load mathlib/anaconda3/conda || echo "module加载失败，可能已在环境中"

echo "激活conda环境..."
conda activate brain-cog

echo "当前Python路径:"
which python
python --version

cd ~/codes/Brain-Cog/examples/Structural_Development/DSD-SNN/cifar100/
echo "进入代码目录完成"

echo "GPU状态："
nvidia-smi

echo "开始运行训练脚本..."
python main_simplified.py --dataset tinyimagenet --num-classes 200 --task_num 40 --output ./output_tinyimagenet
echo "运行训练脚本完成"

date
echo "作业运行完成"