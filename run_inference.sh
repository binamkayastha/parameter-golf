#!/usr/bin/env bash
# Inference launcher - lists models and runs inference

cd "$(dirname "$0")"

echo "Available models:"

models=()
descriptions=()
types=()
i=1

for m in logs/*.npz; do
    if [ -f "$m" ]; then
        size=$(du -h "$m" | cut -f1)
        models+=("$m")
        descriptions+=("$i) $(basename "$m") ($size) [MLX]")
        types+=("mlx")
        i=$((i + 1))
    fi
done

for m in logs/*.ptz; do
    if [ -f "$m" ]; then
        size=$(du -h "$m" | cut -f1)
        models+=("$m")
        descriptions+=("$i) $(basename "$m") ($size) [MLX]")
        types+=("mlx")
        i=$((i + 1))
    fi
done

for m in logs/*.pt; do
    if [ -f "$m" ]; then
        size=$(du -h "$m" | cut -f1)
        models+=("$m")
        descriptions+=("$i) $(basename "$m") ($size) [Torch]")
        types+=("torch")
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
    selected_type="${types[$idx]}"
    read -p "Enter prompt: " prompt
    
    if [ "$selected_type" = "torch" ]; then
        python3 inference_torch.py "$selected" -p "$prompt"
    else
        python3 inference_mlx.py "$selected" -p "$prompt"
    fi
else
    echo "Invalid selection"
    exit 1
fi
