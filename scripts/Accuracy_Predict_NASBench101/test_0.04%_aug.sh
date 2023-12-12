BASE_DIR="."

for PRETRAINED in "nasbench101_latest" "nasbench101_model_best" "nasbench101_model_best_ema"; do

python $BASE_DIR/main.py \
    --dataset nasbench101 \
    --data_path "$BASE_DIR/data/nasbench101/all_nasbench101.pt" \
    --percent 172 \
    --batch_size 2048 \
    --graph_d_model 192 \
    --d_model 192 \
    --graph_d_ff 768 \
    --graph_n_head 6 \
    --depths 6 1 1 1 \
    --lambda_diff 0.2 \
    --save_path "output/nasbench101/neuralformer_0.04%_aug/${PRETRAINED}_test_all/" \
    --pretrained_path "output/nasbench101/neuralformer_0.04%_aug/${PRETRAINED}.pth.tar" \
    --embed_type "nape" \
    --use_extra_token \

done
