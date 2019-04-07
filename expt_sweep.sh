set -e

DATASET_DIR='../../dataset/set1bin'
DATA_DIR='data/set1bin'

expt_dir="${DATA_DIR}/expt"
param_dir="${expt_dir}/param"
sweep_dir="${param_dir}/sweep"

sweeps="1 2 5 10 20"
sweeps="5"

svm_dir="../svm_rank"
svm_learn="${svm_dir}/svm_rank_learn"
svm_classify="${svm_dir}/svm_rank_classify"

for s in $sweeps; do
  echo "#Sweep = ${s}"
  s_dir="${sweep_dir}/${s}"
  for r in $(seq 1 5); do
    echo "Run #${r}"
    run_dir="${s_dir}/${r}"
    log_dir="${run_dir}/log"
    # python -m src.sample_slice -f 0.01 -o 0.8 \
    #   "${DATASET_DIR}/set1bin.train.txt" "${run_dir}/input" > /dev/null
    # for i in 0 1; do
    #   "${svm_learn}" -c 3 "${run_dir}/input/train.slice${i}.txt" \
    #       "${run_dir}/rank${i}.dat" > /dev/null
    #   "${svm_classify}" "${DATASET_DIR}/set1bin.train.txt" \
    #       "${run_dir}/rank${i}.dat" "${run_dir}/score${i}.dat" > /dev/null
    #   python -m src.simulate_click --eta 1 --sweep ${s} --epsilon_p 1 \
    #     --epsilon_n 0.1 "${DATASET_DIR}/set1bin.train.txt" \
    #     "${run_dir}/score${i}.dat" "${log_dir}/log${i}.txt"  > /dev/null
    # done
    python -m src.prop_est --eta 1 -n 10 -a optimizer -m L-BFGS-B \
      "${log_dir}" "${run_dir}/optimizer.txt"
    exit
    # python -m src.prop_est --eta 1 -n 10 -a imp -m L-BFGS-B \
    #   "${log_dir}" "${run_dir}/imp.txt"
    # exit()
    # python -m src.prop_est --eta 1 -n 10 -a naive -m L-BFGS-B \
    #   "${log_dir}" "${run_dir}/naive.txt"
    # python -m src.prop_est --eta 1 -n 10 -a _naive -m L-BFGS-B \
    #   "${log_dir}" "${run_dir}/_naive.txt"
    # python -m src.prop_est --eta 1 -n 10 -a chain -m L-BFGS-B \
    #   "${log_dir}" "${run_dir}/chain.txt"
    # python -m src.prop_est --eta 1 -n 10 -a _chain -m L-BFGS-B \
    #   "${log_dir}" "${run_dir}/_chain.txt"
    # python -m src.prop_est --eta 1 -n 10 -a lsm \
    #   "${log_dir}" "${run_dir}/lsm.txt"
  done
done

# python -m src.eval -k 6 $sweep_dir $sweep_dir Amonut of logged data
