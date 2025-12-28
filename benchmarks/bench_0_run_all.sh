#!/bin/bash

# exit on first error
set -e

################################################################################
# GLobals
START_TIME=$(date)

echo "=============================================================================="
echo " Heavy Tail Flow Benchmark Runner"

################################################################################
# Experiment 1: Linear flow comparison (Synthetic Data)
echo "=============================================================================="
echo "[Experiment 1] Comparison of a Gaussian vs Alpha-stable linear flow on synthetic datasets..."

python bench_1_AlphaStableFlowLinear_comparison.py --config bench_1_config.yaml

echo "[✓] Finished Experiment 1"
echo

################################################################################
# Experiment 2: Alpha influence (Synthetic Data)
echo "=============================================================================="
echo "[Experiment 2] Influence of alpha_model vs alpha_data on synthetic datasets..."

python bench_2_alpha_values_benchmark.py --config bench_2_config.yaml

echo "[✓] Finished Experiment 2"
echo

################################################################################
# Summary
END_TIME=$(date)
echo "=============================================================================="
echo "All experiments completed."
echo "Started at: $START_TIME"
echo "Finished at: $END_TIME"
