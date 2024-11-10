#!/bin/bash

# Set device
DEVICE=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Array of environment names
ENV_NAMES=("Swimmer-v3" "Walker2d-v3" "Ant-v3" "HalfCheetah-v3" "Humanoid-v3" "HumanoidStandup-v2")

# Loop over each environment and run main.py in parallel, saving output and PID to logs
for ENV_NAME in "${ENV_NAMES[@]}"
do
    LOG_FILE="logs/${ENV_NAME}_device${DEVICE}_$(date +'%Y%m%d_%H%M%S').log"
    echo "Running main.py with env_name=${ENV_NAME} on device ${DEVICE}"
    python main.py --rl_type online --env_name "${ENV_NAME}" --device "${DEVICE}" > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "Started ${ENV_NAME} with PID ${PID}. Logging to ${LOG_FILE}."
    echo "PID: ${PID}" >> "${LOG_FILE}"  # Append the PID to the log file
done

# Wait for all background processes to finish
wait

echo "All jobs are completed."
