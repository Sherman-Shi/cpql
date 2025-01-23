#!/bin/bash

# ===========================================
# Parallel Script to Run RL Training Script
# ===========================================

# List of environments to test
ENV_LIST=("Swimmer-v3" "Ant-v3")

# Corresponding CUDA devices for each environment
DEVICE_LIST=(2 2)  # Adjust based on available GPUs

# Number of runs per environment
RUNS=1

# Script to execute
SCRIPT="main.py"

SAMPLER="onestep"
# Output directory
OUTPUT_DIR="results"

# Additional arguments
GROUP_NAME="CPQL"

# Create directories for logs
LOG_DIR="logs"
MAIN_LOG="logs/main_log.txt"
mkdir -p "${LOG_DIR}"
> $MAIN_LOG  # Clear the main log file before starting

# Log header
echo "Running experiments with parallelization..." | tee -a $MAIN_LOG
echo "=========================================" | tee -a $MAIN_LOG

# Loop through each environment and device
for i in "${!ENV_LIST[@]}"; do
    ENV_NAME="${ENV_LIST[$i]}"
    DEVICE="${DEVICE_LIST[$i]}"

    for (( j=1; j<=$RUNS; j++ ))
    do
        # Generate a random seed for each run
        SEED=$(shuf -i 0-99999 -n 1)

        # Log file for this specific run
        LOG_FILE="${LOG_DIR}/${ENV_NAME}_run${j}_seed${SEED}.log"

        # Start the training script in parallel and record its PID
        echo "Starting: Env=${ENV_NAME}, Device=${DEVICE}, Seed=${SEED}, Run=${j}" | tee -a $MAIN_LOG

        python $SCRIPT \
            --env_name $ENV_NAME \
            --device $DEVICE \
            --seed $SEED \
            --dir $OUTPUT_DIR \
            --group $GROUP_NAME \
            --save_checkpoints \
            --quasi_explore True \
            --exploration_sample_num 4 \
            --sampler $SAMPLER \
            --sample_num 1 \
            --num_steps_per_epoch 1000 \
            --online_start_steps 10000 \
            --batch_size 256 \
            --discount 0.99 \
            --TD_sample False \
            --lr_decay \
            > "${LOG_FILE}" 2>&1 &  # Logs output to a file and runs in background

        # Capture and log the PID
        PID=$!
        echo "PID: $PID for ${ENV_NAME} run ${j} (Seed: ${SEED})" | tee -a $MAIN_LOG

        # Brief sleep to prevent overload
        sleep 1
    done
done

# Wait for all parallel jobs to finish
echo "Waiting for all training runs to complete..." | tee -a $MAIN_LOG
wait

echo "All training runs are complete!" | tee -a $MAIN_LOG
