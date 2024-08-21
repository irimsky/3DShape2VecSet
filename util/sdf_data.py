
import os
import glob
import random

import yaml 

import torch
from torch.utils import data

import numpy as np

from PIL import Image

import h5py

class SDFData(data.Dataset):
    def __init__(self, dataset_folder, split, categories=['03001627'], transform=None, sampling=True, num_samples=4096, return_surface=True, surface_sampling=True, pc_size=2048, replica=16):
        
        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.dataset_folder = dataset_folder
        self.point_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_point')
        self.mesh_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_watertight')

        if categories is None:
            categories = os.listdir(self.point_folder)
            categories = [c for c in categories if os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]
        categories.sort()

        # print(categories)

        self.models = []
        # for c_idx, c in enumerate(categories):
        for obj_name in os.listdir(self.dataset_folder):
            subpath = os.path.join(self.dataset_folder, obj_name)
            # assert os.path.isdir(subpath)

            # split_file = os.path.join(subpath, split + '.lst')
            # with open(split_file, 'r') as f:
            #     models_c = f.read().split('\n')
            
            self.models += [
                {
                    'model': obj_name,
                }
            ]

        self.replica = replica

    def __getitem__(self, idx):
        idx = idx % len(self.models)

        # category = self.models[idx]['category']
        model = self.models[idx]['model']
        
        point_path = os.path.join(self.dataset_folder, model, model+'.npz')
        coarse_point_path = os.path.join(self.dataset_folder, model, model+'_32.npz')
        try:
            with np.load(point_path) as data:
                vol_points = data['vol_points'].astype(np.float32)
            with np.load(coarse_point_path) as data:
                coarse_vol_points = data['vol_points'].astype(np.float32)
                
        except Exception as e:
            print(e)
            print(point_path)

        if self.surface_sampling:
            ind = np.random.default_rng().choice(vol_points.shape[0], self.pc_size, replace=False)
            coarse_ind = np.random.default_rng().choice(coarse_vol_points.shape[0], self.pc_size, replace=False)
            vol_points = vol_points[ind]
            coarse_vol_points = coarse_vol_points[coarse_ind]

        # with open(point_path.replace('.npz', '.npy'), 'rb') as f:
        #     scale = np.load(f).item()

        # if self.return_surface:
        #     pc_path = os.path.join(self.mesh_folder, category, '4_pointcloud', model+'.npz')
        #     with np.load(pc_path) as data:
        #         surface = data['points'].astype(np.float32)
        #         surface = surface * scale
        #     if self.surface_sampling:
        #         ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
        #         surface = surface[ind]
        #     surface = torch.from_numpy(surface)

        # if self.sampling:
        #     ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
        #     vol_points = vol_points[ind]
        #     vol_label = vol_label[ind]

        #     ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
        #     near_points = near_points[ind]
        #     near_label = near_label[ind]

        
        vol_points = torch.from_numpy(vol_points)
        coarse_vol_points = torch.from_numpy(coarse_vol_points)
        # vol_label = torch.from_numpy(vol_label).float()

        if self.split == 'train':
            # near_points = torch.from_numpy(near_points)
            # near_label = torch.from_numpy(near_label).float()

            points = vol_points
            coarse_points = coarse_vol_points
            # labels = torch.cat([vol_label, near_label], dim=0)
        else:
            points = vol_points
            coarse_points = coarse_vol_points
            # labels = vol_label

        if self.transform:
            surface, points = self.transform(surface, points)

        return points, coarse_points

        # if self.return_surface:
        #     return points, labels, surface, category_ids[category]
        # else:
        #     return points, labels, category_ids[category]

    def __len__(self):
        if self.split != 'train':
            return len(self.models)
        else:
            return len(self.models) * self.replica