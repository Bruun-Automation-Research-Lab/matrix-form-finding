#!/bin/zsh

# Change to working directory
cd "$HOME/Documents/GitHub_BAR/matrix-form-finding/Lectures" || exit 1

# Activate conda environment (Anaconda)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate CEE6501_jupyter

# Check for input argument
if [[ -z "$1" ]]; then
    echo "Please provide the notebook filename without .ipynb extension"
    exit 1
fi

# Run nbconvert
jupyter nbconvert --to slides "$1.ipynb" --post serve

# Optional: Move up one directory
cd ..
