BASE_DIR="."

python $BASE_DIR/main.py \
    --do_train \
    --device 3 \
    --dataset nasbench101 \
    --data_path "$BASE_DIR/data/nasbench101/all_nasbench101.pt" \
    --percent 4236 \
    --batch_size 128 \
    --graph_d_model 192 \
    --d_model 192 \
    --graph_d_ff 768 \
    --graph_n_head 4 \
    --depths 12 \
    --epochs 3000 \
    --model_ema \
    --lr 1e-4 \
    --lambda_rank 0.2 \
    --test_freq 5 \
    --save_path "output/nasbench101/neuralformer_1%/" \
    --depth_embed --class_token \
    --lambda_consistency 1.0 \
