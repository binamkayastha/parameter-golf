#!/usr/bin/env bash
# Inference launcher - lists models and runs inference

cd "$(dirname "$0")"

echo "Available models:"
models=($(ls -t logs/*.npz 2>/dev/null))

if [ ${#models[@]} -eq 0 ]; then
    echo "No models found in logs/*.npz"
    exit 1
fi

select model in "${models[@]}"; do
    if [ -n "$model" ]; then
        read -p "Enter prompt: " prompt
        python3 inference_mlx.py "$model" -p "$prompt"
        break
    fi
done
