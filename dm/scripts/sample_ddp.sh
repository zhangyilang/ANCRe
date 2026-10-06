# Sample 50K images from DiT checkpoints and compute FID / sFID / IS / precision / recall.
# Usage: bash scripts/sample_ddp.sh <experiment_dir> <model> [extra sample_ddp.py flags]
#   e.g. bash scripts/sample_ddp.sh results/000-DiT-S-2 DiT-S/2
#        bash scripts/sample_ddp.sh results/001-DiT-S-2 DiT-S/2 --use-ancre --ancre-softmax-temp 1e-1
# Requires the ADM reference batch in reference_batch/:
#   https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
EXP_DIR=$1
MODEL=$2
shift 2
MODEL_NAME=${MODEL//\//-}
REF=reference_batch/VIRTUAL_imagenet256_labeled.npz

evaluate() {  # <iter> <cfg-scale>
    torchrun --standalone --nproc_per_node=1 sample_ddp.py --ckpt ${EXP_DIR}/checkpoints/$1.pt --cfg-scale $2 --model ${MODEL} "${EXTRA[@]}"
    python evaluator.py ${REF} samples/${MODEL_NAME}-$1-size-256-vae-ema-cfg-$2-seed-0.npz
    rm samples/${MODEL_NAME}-$1-size-256-vae-ema-cfg-$2-seed-0.npz
}

EXTRA=("$@")
for ITER in {0050000..0400000..50000}; do
    evaluate ${ITER} 1.0
done
evaluate 0400000 1.5
