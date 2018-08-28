set -e

DATASET_DIR='../../dataset/set1bin'
DATA_DIR='data/set1bin'

expt_dir="${DATA_DIR}/expt"
param_dir="${expt_dir}/param"
sweep_dir="${param_dir}/sweep"

sweeps="1 2 5 10 20"

svm_dir="../svm_rank"
svm_learn="${svm_dir}/svm_rank_learn"
svm_classify="${svm_dir}/svm_rank_classify"

# for s in $sweeps; do
#   echo "#Sweep = ${s}"
#   s_dir="${sweep_dir}/${s}"
#   for r in $(seq 6); do
#     echo "Run #${r}"
#     run_dir="${s_dir}/${r}"
#     log_dir="${run_dir}/log"
#     python -m src.sample_slice -f 0.01 -o 0.8 \
#       "${DATASET_DIR}/set1bin.train.txt" "${run_dir}/input"
#     for i in 0 1; do
#       "${svm_learn}" -c 3 "${run_dir}/input/train.slice${i}.txt" \
#           "${run_dir}/rank${i}.dat"
#       "${svm_classify}" "${DATASET_DIR}/set1bin.train.txt" \
#           "${run_dir}/rank${i}.dat" "${run_dir}/score${i}.dat"
#       python -m src.simulate_click --eta 1 --sweep ${s} --epsilon_p 1 \
#         --epsilon_n 0.1 "${DATASET_DIR}/set1bin.train.txt" \
#         "${run_dir}/score${i}.dat" "${log_dir}/log${i}.txt"
#     done
#     python -m src.prop_est --eta 1 -n 10 -a optimizer -m L-BFGS-B \
#       "${log_dir}" "${run_dir}/est.txt"
#   done
# done

python -m src.eval -k 6 $sweep_dir $sweep_dir Amonut of logged data
