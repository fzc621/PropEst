set -e

DATASET_DIR='../../dataset/set1bin'
DATA_DIR='data/set1bin'

expt_dir="${DATA_DIR}/expt"
input_dir="${DATA_DIR}/input"

# sample train slice
python -m src.sample_slice -f 0.01 -o 0.5 \
  "${DATASET_DIR}/set1bin.train.txt" "${input_dir}"
