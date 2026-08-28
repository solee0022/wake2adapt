#!/bin/bash
#!/bin/sh

#SBATCH -J  ASR-Entity-NOISE               
#SBATCH -o  ./out/ASR-Entity-NOISE.%j.out 
#SBATCH -p H200-PCIe-ZT                    
#SBATCH -t 72:00:00                        

## Do not pin a specific node
#SBATCH   --nodes=1

#### Select  GPU
#SBATCH   --gres=gpu:4
#SBTACH   --ntasks=1
# This batch script starts a single python process (= 1 task) that forks 4 GPU workers,
# so cores are reserved via cpus-per-task rather than tasks (4 cores per worker).
#SBATCH   --tasks-per-node=1
#SBATCH   --cpus-per-task=32
#SBATCH -q hpgpu


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
# (With the default, each worker spawns one thread per core and they fight each other
#  -- this is what put 112 threads on 8 cores.)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p ./out

echo "===== speaker x domain x noise(set x SNR x shift) x (asr_adaptation off/on x ref_text) 조합 실행 ====="
# sbatch arguments are forwarded as-is. To split the run across jobs:
#   sbatch scripts/run_all_noise_experiments.sh --speakers P001,P002,P003,P004,P005,P009,P010
#   sbatch scripts/run_all_noise_experiments.sh --speakers P011,P012,P014,P015,P017,P019,P020
# To split by noise condition (result filenames differ per condition, so they never overwrite):
#   sbatch scripts/run_all_noise_experiments.sh --noise-sets Ksponspeech --snrs 1,5
# Once both jobs finish:  python -m src.run_all_noise_experiments --merge-only
python -u -m src.run_all_noise_experiments "$@"

echo "===== Done ====="
