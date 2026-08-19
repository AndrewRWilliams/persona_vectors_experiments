#!/bin/bash
#SBATCH --job-name=cal_projection      # Job name


#salloc --time=1:0:0 --gres=gpu:1  --mem=80G --ntasks=1 --account=def-irina --output=logs/cal_projection_%j.out --error=logs/cal_projection_%j.err

# Load necessary modules (adjust based on your cluster's environment)
#module load python/3.12
module load cudacore/.12.9.1

# Activate your virtual environment if needed
#source ~/envs/persona_vectors/bin/activate
source ~/projects/def-irina/nrjkumar/persona_vectors/llms/bin/activate

# Add the parent directory to PYTHONPATH
export PYTHONPATH="$(dirname $(pwd)):$PYTHONPATH"

# Navigate to the script directory
cd /home/nrjkumar/projects/def-irina/nrjkumar/persona_vectors/scripts

# Run the cal_projection.sh script
bash cal_projection.sh