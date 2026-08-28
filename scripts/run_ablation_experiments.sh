#!/bin/bash
#!/bin/sh

#SBATCH -J  ASR-Entity-ABL                 
#SBATCH -o  ./out/ASR-Entity-ABL.%j.out   
#SBATCH -p A100-80GB                       
#SBATCH -t 72:00:00                        

## Do not pin a specific node
#SBATCH   --nodes=1

#### Select  GPU
#SBATCH   --gres=gpu:4
#SBTACH   --ntasks=1
#SBATCH   --tasks-per-node=4
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

mkdir -p ./out

echo "===== 같은 도메인 ref aud 후보 선택 ablation (length / phoneme / concat) 실행 ====="
# sbatch arguments are forwarded as-is. To split the run across jobs:
#   sbatch scripts/run_ablation_experiments.sh --speakers P001,P002,P003,P004,P005,P009,P010
#   sbatch scripts/run_ablation_experiments.sh --speakers P011,P012,P014,P015,P017,P019,P020
# Once both jobs finish:  python -m src.run_ablation_experiments --merge-only
python -u -m src.run_ablation_experiments "$@"

echo "===== Done ====="
