#!/bin/bash
# Example experiments script for bitter-retrieval

# Basic experiment
python train.py standard_infonce --data data/soft_labels.json --eval-squad

# Hyperparameter sweep
methods=("standard_infonce" "converted_infonce" "kl_soft_infonce")
lrs=(1e-5 2e-5 5e-5)

for method in "${methods[@]}"; do
    for lr in "${lrs[@]}"; do
        echo "Running $method with lr=$lr"
        python train.py $method \
            --data data/soft_labels.json \
            --lr $lr \
            --epochs 2 \
            --eval-squad \
            --wandb-run "${method}_lr${lr}"
    done
done

# KL-specific experiments
for margin in 0.5 1.0 2.0; do
    python train.py kl_soft_infonce \
        --data data/soft_labels.json \
        --margin $margin \
        --teacher-temp 0.01 \
        --student-temp 0.01 \
        --wandb-run "kl_margin${margin}"
done

echo "All experiments completed!" 