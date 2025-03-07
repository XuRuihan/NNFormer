BASE_DIR="."

python $BASE_DIR/main.py \
    --do_train \
    --device 1 \
    --dataset nasbench101 \
    --data_path "$BASE_DIR/data/nasbench101/all_nasbench101.pt" \
    --percent 172 \
    --batch_size 128 \
    --graph_d_model 160 \
    --graph_d_ff 640 \
    --graph_n_head 4 \
    --depths 6 \
    --epochs 3000 \
    --model_ema \
    --lr 1e-4 \
    --lambda_rank 0.2 \
    --test_freq 5 \
    --save_path "output/nasbench101/nnformer_0.04%_l6/" \
    --depth_embed --class_token \
    --lambda_consistency 1.0 \
