#!/bin/bash
#
#SBATCH --job-name=simulate
#SBATCH --output=/home/ulg/PSILab/mvasist/Highres/simulations/output/agg/agg.log
#
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=2-00:00:00
#
#SBATCH --array=1


python /home/mvasist/Highres/simulations/gen.py $SLURM_ARRAY_TASK_ID

###simulations/0000_simulate_%a+320.log