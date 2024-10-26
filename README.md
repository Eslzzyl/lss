This is my fork of [LSS](https://github.com/nv-tlabs/lift-splat-shoot).

clone:
```bash
git clone git@github.com:Eslzzyl/lift-splat-shoot.git -b main
```

Install:

1. Follow the instructions in https://pytorch.org/ to install newest `pytorch` and `torchvision`.

2. Install `nuscenes-devkit`:
    ```bash
    pip install nuscenes-devkit
    ```

    > In order to use `nuscenes-devkit` with Python 3.12, you can use my fork: https://github.com/Eslzzyl/nuscenes-devkit

3. Install other dependencies:
    ```bash
    pip install -r requirements.txt
    ```

Prepare Data:

You should download the nuScenes dataset from its official website and unpack it to a place with enough disk space (no less than 500 GB for `trainval` subset). Theoretically, `mini` subset and `trainval` subset are both supported. I only tested this code for `trainval` subset.

Train:

It is recommended to copy the `train.sh` to a new `my_train.sh` script, and modify `my_train.sh` according to your hardware conditions. Current setting in `train.sh` should be able to train on a 8 * RTX 2080Ti server equipped with no less than 100GB CPU memory. You should also modify the nuScenes dataset root path and TensorBoard log path according to your need.

Then start the training:
```bash
bash my_train.sh
```

> A pretrained EfficientNet checkpoint will be automatically downloaded from pytorch hub. This operation will be performed only once.

Use TensorBoard:
```bash
tensorboard --logdir=./runs
```

Visualize:
```bash
bash visual.sh
```

Evaluate:
```bash
bash eval.sh
```