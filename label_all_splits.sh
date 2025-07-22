#!/bin/bash
# Process all MS MARCO splits with soft labeling
# Usage: ./label_all_splits.sh [model_name] [hf_dataset_name] [dtype] [batch_size]

set -e  # Exit on error

# Default values
MODEL=${1:-"Qwen/Qwen3-4B"}
HF_DATASET=${2:-"nickcdryan/ms_marco_softlabel_Qwen3-4B"}
DTYPE=${3:-"bf16"}
BATCH_SIZE=${4:-"64"}

echo "🚀 Processing ALL MS MARCO splits with soft labeling"
echo "=================================================="
echo "Model: $MODEL"
echo "HF Dataset: $HF_DATASET"
echo "Precision: $DTYPE"
echo "Batch Size: $BATCH_SIZE"
echo ""

# Process each split
for split in train validation test; do
    echo "📊 Processing $split split..."
    poetry run python label_msmarco.py \
        --model "$MODEL" \
        --split "$split" \
        --num 0 \
        --dtype "$DTYPE" \
        --batch-size "$BATCH_SIZE" \
        --upload-hf "$HF_DATASET"
    
    echo "✅ Completed $split split"
    echo ""
done

echo "🎉 ALL SPLITS COMPLETED!"
echo "Dataset available at: https://huggingface.co/datasets/$HF_DATASET" 