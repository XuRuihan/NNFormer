BASE_DIR="."

for PRETRAINED in "nasbench201_latest" "nasbench201_model_best" "nasbench201_model_best_ema"; do

python $BASE_DIR/main.py \
    --dataset nasbench201 \
    --data_path "$BASE_DIR/data/nasbench201/all_nasbench201.pt" \
    --batch_size 2048 \
    --graph_d_model 192 \
    --graph_d_ff 768 \
    --graph_n_head 4 \
    --depths 12 \
    --save_path "output/nasbench201/nnformer_1%/${PRETRAINED}_test_all/" \
    --pretrained_path "output/nasbench201/nnformer_1%/${PRETRAINED}.pth.tar" \
    --depth_embed --class_token \

done
