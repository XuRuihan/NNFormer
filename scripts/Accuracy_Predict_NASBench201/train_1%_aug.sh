BASE_DIR="."

python $BASE_DIR/main.py \
    --do_train \
    --device 1 \
    --dataset nasbench201 \
    --data_path "$BASE_DIR/data/nasbench201/all_nasbench201.pt" \
    --percent 156 \
    --batch_size 128 \
    --graph_d_model 192 \
    --d_model 192 \
    --graph_d_ff 768 \
    --graph_n_head 6 \
    --depths 6 1 1 1 \
    --epochs 4000 \
    --model_ema \
    --lr 1e-4 \
    --lambda_diff 0.2 \
    --test_freq 5 \
    --save_path "output/nasbench201/neuralformer_1%_aug/" \
    --embed_type "nape" \
    --use_extra_token \
    --aug_data_path "$BASE_DIR/data/nasbench201/15625_nasbench201_aug.pt" \
    --lambda_consistency 1.0 \
