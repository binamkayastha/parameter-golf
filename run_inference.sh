#!/usr/bin/env bash
# Inference launcher - lists models and runs inference

cd "$(dirname "$0")"

echo "Available models:"

models=()
descriptions=()
i=1

for m in logs/*.npz; do
    if [ -f "$m" ]; then
        size=$(du -h "$m" | cut -f1)
        models+=("$m")
        descriptions+=("$i) $(basename "$m") ($size)")
        i=$((i + 1))
    fi
done

for m in logs/*.ptz; do
    if [ -f "$m" ]; then
        size=$(du -h "$m" | cut -f1)
        models+=("$m")
        descriptions+=("$i) $(basename "$m") ($size)")
        i=$((i + 1))
    fi
done

if [ ${#models[@]} -eq 0 ]; then
    echo "No models found in logs/"
    exit 1
fi

for desc in "${descriptions[@]}"; do
    echo "$desc"
done

echo ""
read -p "Choose model number: " choice

if [ "$choice" -ge 1 ] && [ "$choice" -le ${#models[@]} ]; then
    idx=$((choice - 1))
    selected="${models[$idx]}"
    read -p "Enter prompt: " prompt
    python3 inference_mlx.py "$selected" -p "$prompt"
else
    echo "Invalid selection"
    exit 1
fi
