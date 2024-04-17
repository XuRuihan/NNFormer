BASE_DIR="."

python $BASE_DIR/main.py \
    --do_train \
    --device 0 \
    --dataset nasbench101 \
    --data_path "$BASE_DIR/data/nasbench101/gates_nasbench101.pt" \
    --percent 73 \
    --batch_size 128 \
    --graph_d_model 128 \
    --graph_d_ff 512 \
    --graph_n_head 4 \
    --depths 6 \
    --epochs 3000 \
    --model_ema \
    --lr 1e-4 \
    --lambda_rank 0.2 \
    --test_freq 5 \
    --save_path "output/gates/nasbench101/neuralformer_1%/" \
    --depth_embed --class_token --avg_tokens \
    --lambda_consistency 1.0 \
