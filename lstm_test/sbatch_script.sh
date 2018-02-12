#!/bin/sh
#SBATCH -t 00:55:00
#SBATCH -J nameOfJob
#SBATCH -A lu2018-2-3
#SBATCH -o stdout_%j.out
#SBATCH -e stderr_%j.err

# shold be lu or gpu
#SBATCH -p gpu

# how many gpus
#SBATCH --gres=gpu:2

#SBATCH --mem-per-cpu=3100

DATA_DIR="$(cat ../data_location.txt)"


echo "start time"
date

cp -r $DATA_DIR $SNIC_TMP

echo "copy done time"
date

python main.py $SNIC_TMP

echo "end time"
date
