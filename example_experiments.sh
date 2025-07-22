#!/bin/bash
# Example experiments script for bitter-retrieval

# Basic experiment
poetry run python -m src.bitter_retrieval.train --method standard_infonce

# Hyperparameter sweep
methods=("standard_infonce" "converted_infonce" "kl_soft_infonce")
lrs=(1e-5 2e-5 5e-5)

for method in "${methods[@]}"; do
    for lr in "${lrs[@]}"; do
        echo "Running $method with lr=$lr"
        poetry run python -m src.bitter_retrieval.train \
            --method $method \
            --learning-rate $lr \
            --num-epochs 2 \
            --run-name "${method}_lr${lr}"
    done
done

# KL-specific experiments
for margin in 0.5 1.0 2.0; do
    poetry run python -m src.bitter_retrieval.train \
        --method kl_soft_infonce \
        --learning-rate 2e-5 \
        --num-epochs 2 \
        --run-name "kl_margin${margin}"
done

echo "All experiments completed!" 