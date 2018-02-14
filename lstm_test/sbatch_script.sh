#!/bin/sh
#SBATCH -t 03:55:00
#SBATCH -J lstm_eeg
#SBATCH -A lu2018-2-3
#// SBATCH -o stdout_%j.out
#// SBATCH -e stderr_%j.err

# shold be lu or gpu
#SBATCH -p gpu

# how many gpus, 2 per node?
#SBATCH --gres=gpu:2

#SBATCH --mem-per-cpu=3100

DATA_DIR="$(cat ../data_location.txt)"
echo "data dir"
echo $DATA_DIR

echo "script"
cat $0

PY_FILE="all_single.py"
echo "py file"
cat $PY_FILE

echo "nvidia smi"
nvidia-smi

echo "start time"
date

cp -r $DATA_DIR $SNIC_TMP
ls $SNIC_TMP
du -h "${SNIC_TMP}/DATA"

echo "copy done time"
date

python $PY_FILE $SNIC_TMP

echo "end time"
date
