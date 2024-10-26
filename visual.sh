#! /bin/bash

CUDA_VISIBLE_DEVICES=0
LOCAL_RANK=0
torchrun --nproc_per_node=1 \
    main.py viz_model_preds trainval \
    --modelf=./ckpts/model40000.pt \
    --dataroot=/root/public/nuScenes/Fulldatasetv1.0 \
    --map_folder=/root/public/nuScenes/Fulldatasetv1.0 \
    --nworkers=8