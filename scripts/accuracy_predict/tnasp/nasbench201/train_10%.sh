BASE_DIR="."

python $BASE_DIR/main.py \
    --do_train \
    --device 3 \
    --dataset nasbench201 \
    --data_path "$BASE_DIR/data/nasbench201/all_nasbench201.pt" \
    --percent 1563 \
    --batch_size 128 \
    --graph_d_model 160 \
    --graph_d_ff 640 \
    --graph_n_head 4 \
    --depths 12 \
    --epochs 3000 \
    --model_ema \
    --lr 1e-4 \
    --lambda_rank 0.2 \
    --test_freq 5 \
    --save_path "output/nasbench201/nnformer_10%/" \
    --depth_embed --class_token \
    --lambda_consistency 1.0 \
