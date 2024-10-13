#! /usr/bin/bash

python main.py train trainval \
    --dataroot=/root/public/nuScenes/Fulldatasetv1.0 \
    --logdir=/root/tf-logs/LSS \
    --gpuid=0 \
    --bsz=16 \
    --nworkers=6 \
    --lr=2e-3
