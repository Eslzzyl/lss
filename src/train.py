"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import os
from time import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from .data import compile_data
from .models import compile_model
from .tools import SimpleLoss, get_batch_iou, get_val_info


local_rank = int(os.environ["LOCAL_RANK"])
dist.init_process_group(backend="nccl")

def train(version,
          dataroot='/data/nuscenes',
          nepochs=1000,

          H=900, W=1600,
          resize_lim=(0.193, 0.225),
          final_dim=(128, 352),
          bot_pct_lim=(0.0, 0.22),
          rot_lim=(-5.4, 5.4),
          rand_flip=True,
          ncams=5,
          max_grad_norm=5.0,
          pos_weight=2.13,
          logdir='./runs',
          ckptdir='./ckpts',

          xbound=[-50.0, 50.0, 0.5],
          ybound=[-50.0, 50.0, 0.5],
          zbound=[-10.0, 10.0, 20.0],
          dbound=[4.0, 45.0, 1.0],

          bsz=16,
          nworkers=4,
          lr=2e-3,
          weight_decay=1e-7,
          ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }
    train_loader, val_loader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          local_rank=local_rank, parser_name='segmentationdata')

    model = compile_model(grid_conf, data_aug_conf, outC=1).cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
        )

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(local_rank)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)

    writer = SummaryWriter(log_dir=logdir)
    val_step = 1000 if version == 'mini' else 10000

    model.train()
    counter = 0
    total_iter = nepochs * len(train_loader)
    t_start = time()
    for epoch in range(nepochs):
        np.random.seed()
        for _, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(train_loader):
            t0 = time()
            opt.zero_grad()
            preds = model(imgs.cuda(local_rank),
                          rots.cuda(local_rank),
                          trans.cuda(local_rank),
                          intrins.cuda(local_rank),
                          post_rots.cuda(local_rank),
                          post_trans.cuda(local_rank),
                          )
            binimgs = binimgs.cuda(local_rank)
            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)
            
            if counter % 100 == 0 and local_rank == 0:
                t_curr = time()
                remaining_iter = total_iter - counter
                remaining_time = (t_curr - t_start) / counter * remaining_iter
                print(f'Epoch: {epoch}, counter: {counter}, loss: {loss.item()}, ETA: {remaining_time / 3600} hours')

            if counter % val_step == 0 and local_rank == 0:
                val_info = get_val_info(model, val_loader, loss_fn, local_rank, use_tqdm=True)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

                model.eval()
                mname = os.path.join(ckptdir, f"model{counter}.pt")
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()
                
        t_curr = time()
        remaining_iter = total_iter - counter
        remaining_time = (t_curr - t_start) / counter * remaining_iter
        print(f'Epoch {epoch} finished, remaining time: {remaining_time / 3600:.2f} hours')
