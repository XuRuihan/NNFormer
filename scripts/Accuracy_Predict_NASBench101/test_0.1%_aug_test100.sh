BASE_DIR="."

for PRETRAINED in "nasbench101_latest" "nasbench101_model_best" "nasbench101_model_best_ema"; do

python $BASE_DIR/main.py \
    --dataset nasbench101 \
    --data_path "$BASE_DIR/data/nasbench101/all_nasbench101.pt" \
    --percent 424 \
    --batch_size 2048 \
    --graph_d_model 192 \
    --d_model 192 \
    --graph_d_ff 768 \
    --graph_n_head 6 \
    --depths 12 \
    --save_path "output/nasbench101/neuralformer_0.1%_aug/${PRETRAINED}_test100/" \
    --pretrained_path "output/nasbench101/neuralformer_0.1%_aug/${PRETRAINED}.pth.tar" \
    --embed_type "nape" \
    --use_extra_token \

done
