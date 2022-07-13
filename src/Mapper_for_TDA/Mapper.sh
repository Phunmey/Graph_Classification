#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for  job on a Compute Canada cluster. 	
# ---------------------------------------------------------------------
#SBATCH --account=def-cakcora
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=7-00:00
#SBATCH --job-name=Mapper-nonsampled
#SBATCH --mail-user=taiwom1@myumanitoba.ca
#SBATCH --mail-type=ALL
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------
# Run your simulation step here...
module load python/3.8
source ~/my_nvenvpy38_test/bin/activate

python ~/projects/def-cakcora/taiwo/Mapper_for_TDA/Mapper_11_5_22.py
# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"

--