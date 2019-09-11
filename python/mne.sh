#!/bin/sh
#SBATCH -t 140:00:00
#SBATCH -J csp_mlp
#SBATCH -A lu2018-2-3
#// SBATCH -o stdout_%j.out
#// SBATCH -e stderr_%j.err

# shold be lu or gpu
#SBATCH -p gpu

# how many gpus, 4 per node, using many seems to crash more often so stick with 1
#SBATCH --gres=gpu:1

# use 5 cores per GPU
#SBATCH -n 5
#SBATCH --mem-per-cpu=3100

source activate mne

DATA_DIR="$(cat ../data_location.txt)"
echo "data dir"
echo $DATA_DIR

echo "script"
cat $0

PY_FILE="test_csp.py"
echo "py file"
cat $PY_FILE

echo "nvidia smi"
nvidia-smi

echo "start time"
date

CURR_DIR="$(pwd)"
cd $DATA_DIR
cp -r --parents DATA/Visual $SNIC_TMP
cp -r --parents DATA/Verbal $SNIC_TMP
cd $CURR_DIR
ls $SNIC_TMP
du -h "${SNIC_TMP}/DATA"

echo "copy done time"
date

python $PY_FILE $SNIC_TMP

echo "end time"
date
