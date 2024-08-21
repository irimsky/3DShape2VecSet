python main_coarse_cond.py \
    --accum_iter 2 \
    --model kl_d512_m512_l8_d24_edm \
    --ae kl_d512_m512_l8 \
    --ae-pth pretrained/ae/kl_d512_m512_l8/checkpoint-199.pth \
    --ae-cond ae_d512_m512 \
    --ae-cond-pth pretrained/ae/ae_d512_m512/checkpoint-199.pth \
    --output_dir output_sn/dm/kl_d512_m512_l8_d24_edm \
    --log_dir output_sn/dm/kl_d512_m512_l8_d24_edm \
    --num_workers 64 \
    --point_cloud_size 2048 \
    --batch_size 2 \
    --epochs 5000 \
    --data_path /data/ljf/shapenet_test \
    --device cuda:4 \
    --save_per_epoch 200 \
    # --resume /data/ljf/3DShape2VecSet/output_coarse_8192/dm/kl_d512_m512_l8_d24_edm/checkpoint-4999.pth \
    

