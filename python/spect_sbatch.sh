#!/bin/sh
#SBATCH -t 60:00:00
#SBATCH -J 2chs5
#SBATCH -A lu2018-2-3
#// SBATCH -o stdout_%j.out
#// SBATCH -e stderr_%j.err

# shold be lu or gpu
#SBATCH -p gpu

# how many gpus, 4 per node, using many seems to crash more often so stick with 1
#SBATCH --gres=gpu:4

# use 5 cores per GPU
#SBATCH -n 20
#SBATCH --mem-per-cpu=3100

DATA_DIR="$(cat ../data_location.txt)"
echo "data dir"
echo $DATA_DIR

echo "script"
cat $0

PY_FILE="conv2d_spect.py"
echo "py file"
cat $PY_FILE

echo "data file"
cat "data.py"

echo "mat file"
cat "../matlab/save transformed/savetf.m"

echo "nvidia smi"
nvidia-smi

echo "start time"
date

cp -r "${DATA_DIR}/DATA" $SNIC_TMP
ls $SNIC_TMP
du -h "${SNIC_TMP}/DATA"

echo "copy done time"
date

matlab -nodisplay -nosplash -nodesktop -r "run('../matlab/save transformed/savetf.m');"
python $PY_FILE $SNIC_TMP

echo "end time"
date
