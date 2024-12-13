#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=16G  # Increased memory
#SBATCH --time=0-12:00
#SBATCH --job-name=translation_job
#SBATCH --output=%j.out
#SBATCH --error=%j.err


# Load necessary modules
module load blis flexiblas arrow/17.0.0 StdEnv/2023 gcc/12.3 openmpi/4.1.5 faiss/1.8.0

# Activate virtual environment
source ~/projects/id/RAGForge/ENV/bin/activate

# Run with time and resource monitoring
python WithRag.py
