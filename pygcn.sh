#!/bin/bash
#SBATCH --job-name="seaice"
#SBATCH --account=def-ka3scott
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --time=10:00:00  # Request 1 hour of compute time
#SBATCH --mem=4GB  # Request 4GB of memory

# salloc --time=2:0:0 --gres=gpu:1 --mem=4GB --ntasks=1 --account=def-ka3scott

source ~/scratch/virenv/bin/activate
module load cuda opencv python scipy-stack StdEnv/2023  gcc/12.3  gdal libspatialindex

# pip install --no-index optuna
# pip install --no-index networkx
# pip install --no-index geopandas
  

# pip install --no-index shapely
# pip install --no-index scikit_image
# pip install --no-index scikit-learn
# pip install --no-index imblearn

# Print GPU information
nvidia-smi

CUDA_LAUNCH_BLOCKING=1 python train.py 

# pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
# pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
# pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
# pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
# pip install torch-geometric