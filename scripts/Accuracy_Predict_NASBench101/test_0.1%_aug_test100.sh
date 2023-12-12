BASE_DIR="/home/disk/NAR-Former"

for PRETRAINED in "nasbench101_latest" "nasbench101_model_best" "nasbench101_model_best_ema"; do

python $BASE_DIR/main.py \
    --dataset nasbench101 \
    --percent 424 \
    --data_path "$BASE_DIR/data/nasbench101/all.pt" \
    --batch_size 1024 \
    --graph_d_model 192 \
    --d_model 192 \
    --graph_d_ff 768 \
    --graph_n_head 6 \
    --depths 6 1 1 1 \
    --lambda_diff 0.2 \
    --save_path "checkpoints_0.1%_aug/${PRETRAINED}_test100/" \
    --pretrained_path "checkpoints_0.1%_aug/${PRETRAINED}.pth.tar" \
    --embed_type "nape" \
    --use_extra_token \
    --aug_data_path "$BASE_DIR/data/nasbench101/OurAug.pt" \
    --lambda_consistency 1.0 \

done