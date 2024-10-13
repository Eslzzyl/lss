This is my fork of [LSS](https://github.com/nv-tlabs/lift-splat-shoot).

clone:
```bash
git clone git@github.com:Eslzzyl/lift-splat-shoot.git -b main
```

Install:
```bash

```

> In order to use `nuscenes-devkit` with 3.12, you can use my fork: https://github.com/Eslzzyl/nuscenes-devkit

Train:
```bash
python main.py train trainval --dataroot=NUSCENES_ROOT --logdir=./runs --gpuid=0
```

Use TensorBoard:
```bash
tensorboard --logdir=./runs --bind_all
```

Visualize:
```bash
python main.py viz_model_preds trainval --modelf=MODEL_LOCATION --dataroot=NUSCENES_ROOT --map_folder=NUSCENES_MAP_ROOT
```

Evaluate:
```bash
python main.py eval_model_iou trainval --modelf=MODEL_LOCATION --dataroot=NUSCENES_ROOT
```