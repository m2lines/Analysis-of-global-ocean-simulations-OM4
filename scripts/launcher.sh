#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=7GB
#SBATCH --begin=now
#SBATCH --time=00:25:00
#SBATCH --job-name=NW2flt

# Usage of the script sbatch --array=0-1900 launcher.sh. In total --array=0-2399 is needed but prohibited by scheduling system

echo " "
scontrol show jobid -dd $SLURM_JOB_ID
echo " "
echo "The number of alphafold processes:"
ps -e | grep -i alphafold | wc -l
echo " "
module purge

# Here we calculate time_idx from the array id number
time_idx=$(( SLURM_ARRAY_TASK_ID))

singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u filter-interfaces-GM-filter.py --time_idx=${time_idx} "
