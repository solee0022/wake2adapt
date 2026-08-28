# ASR-Entity

## SLURM job scripts

Uses the `asr-entity` conda environment, with Qwen2.5-Omni-7B as the ASR model.

### Quick runs / debugging on a single speaker-domain pair

- Use a single GPU (`--gres=gpu:1`) to quickly verify one speaker/domain combination
- Before running, edit the hardcoded `SPEAKER=`, `DOMAIN=`, `REF_AUD=`, and `REF_TEXT=` values at the top of the script to the combination you want

#### 1. Grab a GPU with `srun` and run it directly in an interactive shell
```bash
srun --gres=gpu:1 --cpus-per-gpu=8 -p A100-80GB -q hpgpu --pty bash
# once the GPU is allocated, run it right there
conda activate asr-entity
python infer_qwen2.5_omni_batch.py \
    --speaker P001 \
    --domain roads \
    --ref_aud RD16245_P001.wav \
    --ref_text "상곡안길" \
    --asr_adaptation \
    --batch_size 8
```

#### 2. Queue it in the background with `sbatch` instead of holding a GPU yourself
```bash
sbatch run_infer.sh
```

### Running every combination at once (`run_all_experiments.sh`)
```bash
sbatch run_all_experiments.sh
```
