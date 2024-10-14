#! /usr/bin/bash

OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 \
    main.py train trainval \
    --dataroot=/root/public/nuScenes/Fulldatasetv1.0 \
    --logdir=/root/tf-logs/LSS \
    --gpuid=0 \
    --bsz=16 \
    --nworkers=4 \
    --lr=2e-3
