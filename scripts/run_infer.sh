#!/bin/bash
#!/bin/sh

#SBATCH -J  ASR-Entity                     
#SBATCH -o  ./out/ASR-Entity.%j.out       
#SBATCH -p A100-80GB                       
#SBATCH -t 72:00:00                       

## Do not pin a specific node
#SBATCH   --nodes=1

#### Select  GPU
#SBATCH   --gres=gpu:1
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

# ===== Run settings =====
SPEAKER=P001
DOMAIN=roads
REF_AUD=pr0131_pr_P001.wav
REF_TEXT="안드로이드"
BATCH_SIZE=8

echo "===== [1/2] ASR (zero-shot) + phonetic edit distance retrieval ====="
python -u src/infer_qwen2.5_omni_batch.py \
    --speaker $SPEAKER \
    --domain $DOMAIN \
    --batch_size $BATCH_SIZE

echo "===== [2/2] ASR adaptation (1-shot) + phonetic edit distance retrieval ====="
python -u src/infer_qwen2.5_omni_batch.py \
    --speaker $SPEAKER \
    --domain $DOMAIN \
    --ref_aud $REF_AUD \
    --ref_text "$REF_TEXT" \
    --asr_adaptation \
    --batch_size $BATCH_SIZE

echo "===== Done ====="



