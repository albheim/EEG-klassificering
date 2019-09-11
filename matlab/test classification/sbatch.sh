#!/bin/sh
#SBATCH -t 130:00:00
#SBATCH -J mat_ra1
#SBATCH -A lu2018-2-3

# should be lu or gpu
#SBATCH -p gpu

#SBATCH --gres=gpu:4

#SBATCH -n 20
#SBATCH --mem-per-cpu=3100

echo "start time"
date

matlab -nodisplay -nosplash -nodesktop -r "run('main.m');"

echo "end time"
date
