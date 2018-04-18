#!/bin/sh
#SBATCH -t 00:05:00
#SBATCH -J lstm_eeg
#SBATCH -A lu2018-2-3
#// SBATCH -o stdout_%j.out
#// SBATCH -e stderr_%j.err

# shold be lu or gpu
#SBATCH -p gpu

# how many gpus, 4 per node
#SBATCH --gres=gpu:1

#SBATCH --mem-per-cpu=3100

echo "script"
cat $0

echo "nvidia smi"
nvidia-smi

echo $SNIC_SITE
echo $SNIC_RESOURCE

echo "asdf %j sdf"
