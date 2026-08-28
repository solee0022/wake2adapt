#!/bin/bash
#!/bin/sh

#SBATCH -J  ASR-Entity-ALL                 
#SBATCH -o  ./out/ASR-Entity-ALL.%j.out    
#SBATCH -p RTX6000ADA                    
#SBATCH -t 72:00:00                        

## Do not pin a specific node
#SBATCH   --nodes=1

#### Select  GPU
#SBATCH   --gres=gpu:4
#SBTACH   --ntasks=1
# This batch script starts a single python process (= 1 task) that forks 4 GPU workers,
# so cores are reserved via cpus-per-task rather than tasks (4 cores per worker).
#SBATCH   --tasks-per-node=1
#SBATCH   --cpus-per-task=16
##SBATCH -q hpgpu


cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

# path  Erase because of the crash
module  purge

echo "source  $HOME/miniforge3/etc/profile.d/conda.sh"
source  $HOME/miniforge3/etc/profile.d/conda.sh

conda activate asr-entity

# The 4 GPU workers share cpus-per-task, so cap the threads per process.
# (With the default, each worker spawns one thread per core and they fight each other)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p ./out

echo "===== speaker x domain x (asr_adaptation off/on x ref_text) 전체 조합 실행 ====="
# sbatch arguments are forwarded as-is. To split the run across jobs:
#   sbatch scripts/run_all_experiments.sh --speakers P001,P002,P003,P004,P005,P009,P010
# Once both jobs finish:  python -m src.run_all_experiments --merge-only
python -u -m src.run_all_experiments "$@"

echo "===== Done ====="
