set -e

DATASET_DIR='../../dataset/set1bin'
DATA_DIR='data/set1bin'

expt_dir="${DATA_DIR}/expt"
input_dir="${DATA_DIR}/input"
model_dir="${expt_dir}/model"
log_dir="${expt_dir}/log"

python -m src.prop_est --eta 1 -n 10 "${log_dir}" "${expt_dir}/result.txt"
