set -e

DATASET_DIR='../../dataset/set1bin'
DATA_DIR='data/set1bin'

expt_dir="${DATA_DIR}/expt"
input_dir="${DATA_DIR}/input"
model_dir="${expt_dir}/model"
log_dir="${expt_dir}/log"

# python -m src.prop_est --eta 1 -n 10 -a naive "${log_dir}" "${expt_dir}/result_n.txt"

# python -m src.prop_est --eta 1 -n 10 -a chain "${log_dir}" "${expt_dir}/result_c.txt"

# python -m src.prop_est --eta 1 -n 10 -a optimizer -m L-BFGS-B "${log_dir}" "${expt_dir}/result_o_l.txt"

python -m src.prop_est --eta 1 -n 10 -a optimizer -l -m L-BFGS-B "${log_dir}" "${expt_dir}/result_o_l_mid.txt"

# python -m src.prop_est --eta 1 -n 10 -a optimizer -m TNC "${log_dir}" "${expt_dir}/result_o_tnc.txt"

# python -m src.prop_est --eta 1 -n 10 -a optimizer -m SLSQP "${log_dir}" "${expt_dir}/result_o_slsqp.txt"
