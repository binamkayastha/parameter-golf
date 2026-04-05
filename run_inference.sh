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

# Check for final_model.pt in current directory
if [ -f "final_model.pt" ]; then
    size=$(du -h final_model.pt | cut -f1)
    models+=("final_model.pt")
    descriptions+=("$i) final_model.pt ($size) [Torch - current dir]")
    types+=("torch")
    i=$((i + 1))
fi

if [ -f "final_model.int8.ptz" ]; then
    size=$(du -h final_model.int8.ptz | cut -f1)
    models+=("final_model.int8.ptz")
    descriptions+=("$i) final_model.int8.ptz ($size) [Torch - current dir]")
    types+=("torch")
    i=$((i + 1))
fi

if [ ${#models[@]} -eq 0 ]; then
    echo "No models found in logs/ or current directory"
    echo ""
    echo "After running train_gpt.py, models are saved as:"
    echo "  - final_model.pt"
    echo "  - final_model.int8.ptz"
    echo ""
    echo "Move them to logs/ or run inference directly:"
    echo "  python3 inference_torch.py final_model.pt -p \"hello\""
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
