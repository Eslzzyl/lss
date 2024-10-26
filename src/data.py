"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()

        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    
    def get_scenes(self):
        '''
        nuScenes 存在 mini 和 trainval 两个子集。这个函数的作用是根据 self.nusc.version 和 self.is_train 来返回对应的子集中的 scenes
        '''
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [s for s in self.nusc.sample]

        # 去除不在当前子集（mini 或 trainval）中的 samples
        samples = [s for s in samples if self.nusc.get('scene', s['scene_token'])['name'] in self.scenes]

        # 按照 scene_token 和 timestamp 排序
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]) else True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        imgs = []
        rotations = []
        translations = []
        camera_intrinsics = []
        post_rotations = []
        post_translations = []
        for cam in cams:
            sample = self.nusc.get('sample_data', rec['data'][cam])
            img_name = os.path.join(self.nusc.dataroot, sample['filename'])
            img = Image.open(img_name)
            post_rotation = torch.eye(2)
            post_translation = torch.zeros(2)

            sensors = self.nusc.get('calibrated_sensor', sample['calibrated_sensor_token'])
            # 相机内参
            camera_intrinsic = torch.Tensor(sensors['camera_intrinsic'])
            # 相机位姿
            rotation = torch.Tensor(Quaternion(sensors['rotation']).rotation_matrix)
            # 相机位移
            translation = torch.Tensor(sensors['translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, aug_post_rotation, aug_post_translation = img_transform(img, post_rotation, post_translation,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )
            
            # for convenience, make augmentation matrices 3x3
            post_translation = torch.zeros(3)
            post_rotation = torch.eye(3)
            post_translation[:2] = aug_post_translation
            post_rotation[:2, :2] = aug_post_rotation

            imgs.append(normalize_img(img))
            camera_intrinsics.append(camera_intrinsic)
            rotations.append(rotation)
            translations.append(translation)
            post_rotations.append(post_rotation)
            post_translations.append(post_translation)

        return (torch.stack(imgs), torch.stack(rotations), torch.stack(translations),
                torch.stack(camera_intrinsics), torch.stack(post_rotations), torch.stack(post_translations))

    def get_lidar_data(self, rec, nsweeps):
        # Returned tensor is 5(x, y, z, reflectance, dt) x N
        pts = get_lidar_data(self.nusc, rec, nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # reflectance and dt are not used

    def get_binimg(self, rec):
        """
        获取二值图像。

        参数:
        rec (dict): 包含 annotations 和传感器数据的记录。

        返回:
        torch.Tensor: 生成的二值图像，形状为 (1, nx[0], nx[1])。

        说明:
        该方法从记录中提取 annotations，并根据自车位姿对每个 annotations 进行变换。然后，将每个 annotations 的底部角点转换为图像坐标，并在图像中填充相应的多边形区域，生成二值图像。

        这个函数主要用于将三维空间中的车辆边界框投影到二维平面上。
        """
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(ego_pose['translation'])
        rot = Quaternion(ego_pose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0)

    def choose_cams(self):
        # self.data_aug_conf['cams'] 来自 train.py 中的配置，它的值是 ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        # 即，self.data_aug_conf['cams'] 是一个长度为 6 的列表
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'], replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}. \n Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    '''
    用于测试和验证的数据集

    和 SegmentationData 的唯一不同是多返回了 lidar_data
    '''
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]
        
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    '''
    用于训练的数据集

    和 VizData 的唯一不同是没有返回 lidar_data
    '''
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


def worker_random_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz, nworkers, local_rank, parser_name):
    nusc = NuScenes(version=f'v1.0-{version}', dataroot=dataroot, verbose=True)
    parser = {
        'vizdata': VizData,     # 用于测试和验证
        'segmentationdata': SegmentationData,   # 用于训练
    }[parser_name]
    train_dataset = parser(nusc, is_train=True, data_aug_conf=data_aug_conf, grid_conf=grid_conf)
    val_dataset = parser(nusc, is_train=False, data_aug_conf=data_aug_conf, grid_conf=grid_conf)

    # 分布式训练需要使用自定义的分布式采样器
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              sampler=train_sampler,    # 分布式采样器，注意自定义采样器与shuffle不能同时使用
                                              worker_init_fn=worker_random_init)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return train_loader, val_loader
