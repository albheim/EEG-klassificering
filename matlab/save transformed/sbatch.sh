#!/bin/sh
#SBATCH -t 30:00:00
#SBATCH -J matlab
#SBATCH -A lu2018-2-3

# should be lu or gpu
#SBATCH -p gpu

#SBATCH --gres=gpu:4

#SBATCH --tasks-per-node 20
#SBATCH --mem-per-cpu=3100

echo "start time"
date

matlab -nodisplay -nosplash -nodesktop -r "run('savetf.m');"

echo "end time"
date
