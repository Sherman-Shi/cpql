#!/bin/bash

# =====================================
# Visualization Runner Script
# =====================================
# Runs visualize.py for multiple iterations with varying `sample_num` values.
# Allows setting `env_name` and `dir` as inputs.
# Repeats the sampling process 10 times for each configuration.
# =====================================

# Input Parameters
ENV_NAME=${1:-"Swimmer-v3"}  # Default to Humanoid-v3 if not provided
CHECKPOINT_DIR=${2:-"/home/sherman/Desktop/Consistency/results/online/Swimmer-v3/QL|15|alpha-0.05|eta-1.0|sampler_onestep_quasi_monte_carlo|test_qnorm/"}  # Default checkpoint directory
DEVICE=${3:-0}             # Default CUDA device 0
SAMPLER=${4:-"onestep_monte_carlo"}  # Default sampler
TD_SAMPLE=${5:-False}       # Default TD_sample to False
GROUP=${6:-"Quasi-CPQL-dev"} # Default group name for logging

# Sample numbers to test
SAMPLE_NUMS=(512 512 512 512 512 1024 1024)

# Directory to store logs
LOG_DIR="logs_visualization"
mkdir -p "${LOG_DIR}"

# Repeat each sample_num configuration 10 times
REPEAT=1  # Number of repetitions for each sample_num

# Iterate over sample_num values
for SAMPLE_NUM in "${SAMPLE_NUMS[@]}"
do
    for ((i=1; i<=REPEAT; i++))  # Repeat 10 times for each sample_num
    do
        # Generate a random seed for each run
        SEED=$(shuf -i 0-99999 -n 1)

        # Log file for the current run
        LOG_FILE="${LOG_DIR}/${ENV_NAME}_${SAMPLER}_samplenum${SAMPLE_NUM}_seed${SEED}_run${i}.log"

        echo "Starting visualization (run ${i}) with sample_num=${SAMPLE_NUM}, seed=${SEED} on device=${DEVICE}"
        echo "Logging output to ${LOG_FILE}"

        # Run the visualization script
        python visualize.py \
            --env_name "${ENV_NAME}" \
            --dir "${CHECKPOINT_DIR}" \
            --device "${DEVICE}" \
            --sampler "${SAMPLER}" \
            --sample_num "${SAMPLE_NUM}" \
            --TD_sample "${TD_SAMPLE}" \
            --seed "${SEED}" \
            --group "${GROUP}" > "${LOG_FILE}" 2>&1 &

        # Wait a short time to avoid system overload
        sleep 1
    done
done

# Wait for all background processes to complete
echo "Waiting for all visualization jobs to complete..."
wait

echo "All visualization jobs have completed successfully."
