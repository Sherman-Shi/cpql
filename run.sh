#!/bin/bash

# =====================================
# RL Experiment Runner Script
# =====================================
# This script runs main.py for multiple environments in parallel,
# logs the output, and tracks the processes using PIDs.
# =====================================

# Set the default device
DEVICE=0

# Directory to store logs
LOG_DIR="logs"

# Create logs directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Array of environment names
ENV_NAMES=("Swimmer-v3" "Walker2d-v3" "Ant-v3" "HalfCheetah-v3" "Humanoid-v3" "HumanoidStandup-v2")

# Algorithm name
ALGO="CPQL"

# Timestamp for log files
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# Iterate over each environment and launch main.py
for ENV_NAME in "${ENV_NAMES[@]}"
do
    # Define log file for the current environment
    LOG_FILE="${LOG_DIR}/${ENV_NAME}_device${DEVICE}_${TIMESTAMP}.log"

    echo "Starting main.py for environment=${ENV_NAME} on device=${DEVICE}"
    echo "Logging output to ${LOG_FILE}"

    # Launch main.py in the background and redirect output to log file
    python main.py --group "${ALGO}" --rl_type online --env_name "${ENV_NAME}" --device "${DEVICE}" > "${LOG_FILE}" 2>&1 &

    # Capture PID of the background process
    PID=$!
    echo "Started ${ENV_NAME} with PID ${PID}. Logging to ${LOG_FILE}."
    echo "PID: ${PID}" >> "${LOG_FILE}"  # Append PID to log file
done

# Wait for all background processes to complete
echo "Waiting for all jobs to complete..."
wait

echo "All jobs have completed successfully."
