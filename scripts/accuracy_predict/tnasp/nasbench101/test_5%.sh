BASE_DIR="."

for PRETRAINED in "nasbench101_latest" "nasbench101_model_best" "nasbench101_model_best_ema"; do

python $BASE_DIR/main.py \
    --dataset nasbench101 \
    --data_path "$BASE_DIR/data/nasbench101/all_nasbench101.pt" \
    --percent 21180 \
    --batch_size 2048 \
    --graph_d_model 192 \
    --graph_d_ff 768 \
    --graph_n_head 4 \
    --depths 12 \
    --save_path "output/nasbench101/nnformer_5%/${PRETRAINED}_test_all/" \
    --pretrained_path "output/nasbench101/nnformer_5%/${PRETRAINED}.pth.tar" \
    --depth_embed --class_token \

done
