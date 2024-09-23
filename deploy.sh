#!/bin/bash

# Check the number of parameters
if [ "$#" -ne 1 ]; then
    echo "Error: Exactly one argument is required."
    echo "Available option: dg."
    exit 1
fi

# Activate the virtual environment
conda activate streamer-sales

# Optional GPU configuration
export HF_ENDPOINT="https://hf-mirror.com"

# Start the digital human service
if [ "$1" == "dg" ]; then
    echo "Starting Digital Human Service..."
    uvicorn server.digital_human.digital_human_server:app --host 0.0.0.0 --port 8002
else
    echo "Error: Unsupported parameter '$1'."
    echo "Available option: dg."
    exit 1
fi

exit 0