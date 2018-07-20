set -e

DATASET_DIR='../../dataset/set1bin'
DATA_DIR='data/set1bin'

expt_dir="${DATA_DIR}/expt"
input_dir="${DATA_DIR}/input"
model_dir="${expt_dir}/model"


for i in 0 1;
do
  python -m src.simulate_click --eta 1 --epsilon_p 1 --epsilon_n 0.1 \
    "${DATASET_DIR}/set1bin.train.txt" "${model_dir}/score${i}.dat" \
    "${expt_dir}/log${i}.txt"
done
