#!/bin/bash

# =====================================
# Parallel Experiment Runner Script
# =====================================
# Runs main.py with specified parameters across multiple CUDA devices,
# and uses random seeds for each run.
# =====================================

# Set the devices to use (modify as needed)
DEVICES=(0 1 2 3)

# Directory to store logs
LOG_DIR="logs"

# Create logs directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Environment name
ENV_NAME="HumanoidStandup-v2"

# Parameter configurations
CONFIGS=(
    "Monte-CPQL-dev onestep_monte_carlo 5"
    "Quasi-CPQL-dev onestep_quasi_monte_carlo 5"
    "CPQL onestep 1"
)

# Timestamp for log files
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# Device counter
DEVICE_COUNT=${#DEVICES[@]}
DEVICE_IDX=3

# Iterate over configurations and launch jobs on different devices
for CONFIG in "${CONFIGS[@]}"
do
    # Parse configuration
    read -r ALGO SAMPLER SAMPLE_NUM <<< "${CONFIG}"

    # Select the current device in a round-robin fashion
    DEVICE=${DEVICES[DEVICE_IDX]}

    # Generate a random seed
    SEED=$(shuf -i 0-99999 -n 1)  # Generates a random seed between 0 and 99999

    # Define log file for the current configuration
    LOG_FILE="${LOG_DIR}/${ENV_NAME}_${ALGO}_${SAMPLER}_sample${SAMPLE_NUM}_seed${SEED}_device${DEVICE}_${TIMESTAMP}.log"

    echo "Starting main.py for ALGO=${ALGO}, SAMPLER=${SAMPLER}, SAMPLE_NUM=${SAMPLE_NUM}, seed=${SEED} on device=${DEVICE}"
    echo "Logging output to ${LOG_FILE}"

    # Launch main.py in the background and redirect output to log file
    python main.py --group "${ALGO}" --rl_type online --env_name "${ENV_NAME}" \
        --device "${DEVICE_IDX}" --sampler "${SAMPLER}" \
        --sample_num "${SAMPLE_NUM}" --seed "${SEED}" > "${LOG_FILE}" 2>&1 &

    # Capture PID of the background process
    PID=$!
    echo "Started ALGO=${ALGO}, SAMPLER=${SAMPLER}, SAMPLE_NUM=${SAMPLE_NUM} with seed=${SEED} and PID=${PID} on device=${DEVICE}. Logging to ${LOG_FILE}."
    echo "PID: ${PID}" >> "${LOG_FILE}"  # Append PID to log file

    # Update device index for the next job
    DEVICE_IDX=$(( (DEVICE_IDX + 1) % DEVICE_COUNT ))
done

# Wait for all background processes to complete
echo "Waiting for all jobs to complete..."
wait

echo "All jobs have completed successfully."
