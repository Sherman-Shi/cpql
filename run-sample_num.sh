#!/bin/bash

# =====================================
# Parallel Experiment Runner Script
# =====================================
# Runs main.py with sampler=onestep_quasi_monte_carlo,
# varying `sample_num` values across multiple CUDA devices,
# and using random seeds for each run.
# =====================================

# Set the devices to use (modify as needed)
DEVICES=(0 1 2 3)

# Directory to store logs
LOG_DIR="logs"

# Create logs directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Array of sample numbers to test
SAMPLE_NUMS=(5 10 20 50)

# Environment name
ENV_NAME="HalfCheetah-v3"

# Algorithm name
ALGO="Monte-CPQL-dev"
SAMPLER="onestep_monte_carlo"

# Timestamp for log files
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# Device counter
DEVICE_COUNT=${#DEVICES[@]}
DEVICE_IDX=0

# Iterate over sample numbers and launch jobs on different devices
for SAMPLE_NUM in "${SAMPLE_NUMS[@]}"
do
    # Select the current device in a round-robin fashion
    DEVICE=${DEVICES[DEVICE_IDX]}

    # Generate a random seed
    SEED=$(shuf -i 0-99999 -n 1)  # Generates a random seed between 0 and 99999

    # Define log file for the current configuration
    LOG_FILE="${LOG_DIR}/${ENV_NAME}_sample${SAMPLE_NUM}_seed${SEED}_device${DEVICE}_${TIMESTAMP}.log"

    echo "Starting main.py for sample_num=${SAMPLE_NUM}, seed=${SEED} on device=${DEVICE}"
    echo "Logging output to ${LOG_FILE}"

    # Launch main.py in the background and redirect output to log file
    python main.py --group "${ALGO}" --rl_type online --env_name "${ENV_NAME}" \
        --device "${DEVICE}" --sampler "${SAMPLER}" \
        --sample_num "${SAMPLE_NUM}" --seed "${SEED}" > "${LOG_FILE}" 2>&1 &

    # Capture PID of the background process
    PID=$!
    echo "Started sample_num=${SAMPLE_NUM} with seed=${SEED} and PID=${PID} on device=${DEVICE}. Logging to ${LOG_FILE}."
    echo "PID: ${PID}" >> "${LOG_FILE}"  # Append PID to log file

    # Update device index for the next job
    DEVICE_IDX=$(( (DEVICE_IDX + 1) % DEVICE_COUNT ))
done

# Wait for all background processes to complete
echo "Waiting for all jobs to complete..."
wait

echo "All jobs have completed successfully."
