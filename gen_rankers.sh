set -e

DATASET_DIR='../../dataset/set1bin'
DATA_DIR='data/set1bin'

expt_dir="${DATA_DIR}/expt"
input_dir="${DATA_DIR}/input"
model_dir="${expt_dir}/model"

svm_dir='../svm_rank'

# sample train slice
python -m src.sample_slice -f 0.01 -o 0.5 \
  "${DATASET_DIR}/set1bin.train.txt" "${input_dir}"

# train two rankers
svm_learn="${svm_dir}/svm_rank_learn"
svm_classify="${svm_dir}/svm_rank_classify"
for i in 0 1;
do
  "${svm_learn}" -c 3 "${input_dir}/train.slice${i}.txt" \
      "${model_dir}/rank${i}.dat"
  "${svm_classify}" "${DATASET_DIR}/set1bin.train.txt" \
      "${model_dir}/rank${i}.dat" "${model_dir}/score${i}.dat"
done
