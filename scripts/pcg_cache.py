import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import time
import random
import cv2
import os
import torch
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--terrain', type=str, required=True, help='directory path to terrain dataset')
    parser.add_argument('--outdir', type=str, required=True)
    assert os.path.exists("./scenedreamer_released.pt")
    pcg_asset = torch.load("./scenedreamer_released.pt", map_location='cpu')
    args = parser.parse_args()
    terrain_dir = args.terrain
    outdir = args.outdir
    sample_height = 256
    sample_size = 1024
    os.makedirs(outdir, exist_ok=True)

    trees_models = pcg_asset['assets']

    # can be customized
    biome_trees_dict = {
        'desert': [],
        'savanna': [5],
        'twoodland': [1, 7],
        'tundra': [],
        'seasonal forest': [1, 2],
        'rainforest': [1, 2, 3],
        'temp forest': [4],
        'temp rainforest': [0, 3],
        'boreal': [5,6,7],
        'water': [],
    }

    biome2mclabels = torch.tensor([28, 9, 8, 1, 9, 8, 9, 8, 30, 26], dtype=torch.int32)
    biome_names = list(biome_trees_dict.keys())
    chunk_grid_x, chunk_grid_y = torch.meshgrid(torch.arange(sample_size), torch.arange(sample_size))

    terrain_list = os.listdir(terrain_dir)
    for world in tqdm(terrain_list):
        voxel_t = torch.zeros(sample_height, sample_size, sample_size).to(torch.int32)
        current_dir = os.path.join(terrain_dir, world)
        height_map = np.load(os.path.join(current_dir, 'biome_rivers_height.npy'))
        height_map[height_map < 0] = 0
        height_map = ((height_map - height_map.min()) / (1 - height_map.min()) * (sample_height - 1)).astype(np.int16)
        semantic_map = cv2.imread(os.path.join(current_dir, 'biome_rivers_labels.png'), 0)
        tree_map = cv2.imread(os.path.join(current_dir, 'biome_trees_dist.png'), 0)
        total_size = height_map.shape[0]
        crop_pos_x, crop_pos_y = np.random.randint(0, total_size - sample_size, size=2)
        org_height_map = height_map[crop_pos_x: crop_pos_x + sample_size, crop_pos_y: crop_pos_y + sample_size].astype(int)
        chunk_height_map = torch.from_numpy(org_height_map)[None, ...]
        chunk_semantic_map = semantic_map[crop_pos_x: crop_pos_x + sample_size, crop_pos_y: crop_pos_y + sample_size]
        chunk_trees_map = tree_map[crop_pos_x: crop_pos_x + sample_size, crop_pos_y: crop_pos_y + sample_size]
        org_semantic_map = torch.from_numpy(chunk_semantic_map.copy())
        org_semantic_map[chunk_trees_map != 255] = 10
        chunk_semantic_map = biome2mclabels[torch.from_numpy(chunk_semantic_map)[None, ...].long().contiguous()]
        voxel_t = voxel_t.scatter_(0, chunk_height_map, chunk_semantic_map)
        for preproc_step in range(8):
            voxel_t = voxel_t.scatter(0, torch.clip(chunk_height_map + preproc_step + 1, 0, sample_height - 1), chunk_semantic_map)

        chunk_height_map = chunk_height_map + 8
        chunk_height_map = chunk_height_map[0]
        boundary_detect = 50
        for biome_id in range(biome2mclabels.shape[0]):
            tree_pos_mask = (chunk_trees_map == biome_id)
            tree_pos_x = chunk_grid_x[tree_pos_mask]
            tree_pos_y = chunk_grid_y[tree_pos_mask]
            tree_pos_h = chunk_height_map[tree_pos_mask]
            assert len(tree_pos_x) == len(tree_pos_y)
            selected_trees = biome_trees_dict[biome_names[biome_id]]
            if len(selected_trees) == 0:
                continue
            for idx in range(len(tree_pos_x)):
                if tree_pos_x[idx] < boundary_detect or tree_pos_x[idx] > sample_size - boundary_detect or tree_pos_y[idx] < boundary_detect or tree_pos_y[idx] > sample_size - boundary_detect or tree_pos_h[idx] > sample_height - boundary_detect:
                    # FIXME: hack, to avoid out of index near the boundary
                    continue
                tree_id = random.choice(selected_trees)
                tmp = voxel_t[tree_pos_h[idx]: tree_pos_h[idx] + trees_models[tree_id].shape[0], tree_pos_x[idx]: tree_pos_x[idx] + trees_models[tree_id].shape[1], tree_pos_y[idx]: tree_pos_y[idx] + trees_models[tree_id].shape[2]]
                tmp_mask = (tmp == 0)
                try:
                    voxel_t[tree_pos_h[idx]: tree_pos_h[idx] + trees_models[tree_id].shape[0], tree_pos_x[idx]: tree_pos_x[idx] + trees_models[tree_id].shape[1], tree_pos_y[idx]: tree_pos_y[idx] + trees_models[tree_id].shape[2]][tmp_mask] = trees_models[tree_id][tmp_mask]
                except:
                    print('height?', tree_pos_h[idx])
                    print(tmp_mask.shape)
                    print(tmp.shape)
                    print(trees_models[tree_id].shape)
                    print(voxel_t.shape)
                    print(tree_id)
                    raise NotImplementedError
        trans_mat = torch.eye(4)  # Transform voxel to world
        # Generate heightmap for camera trajectory generation
        m, h = torch.max((torch.flip(voxel_t, [0]) != 0).int(), dim=0, keepdim=False)
        heightmap = voxel_t.shape[0] - 1 - h
        heightmap[m == 0] = 0  # Special case when the whole vertical column is empty
        voxel_t = voxel_t.numpy()
        voxel_value = voxel_t[voxel_t != 0]
        voxel_x, voxel_y, voxel_z = np.where(voxel_t != 0)
        current_height_map = (chunk_height_map / (sample_height - 1))[None, None, ...]
        current_semantic_map = F.one_hot(org_semantic_map.to(torch.int64)).to(torch.float).permute(2, 0, 1)[None, ...]
        semantic_map = torch.argmax(current_semantic_map, dim=1)
        print('semantic map after one hot and argmax', torch.unique(semantic_map, return_counts=True))
        print(current_height_map.shape)
        print(current_semantic_map.shape)
        print(heightmap.shape)
        print(voxel_t.shape)
        print(voxel_value.shape)
        print(voxel_x.shape)
        print(voxel_y.shape)
        print(voxel_z.shape)
        print(voxel_z.dtype)
        voxel_sparse = np.stack([voxel_x, voxel_y, voxel_z, voxel_value])
        print(voxel_sparse.shape)
        current_outdir = os.path.join(outdir, world)
        os.makedirs(current_outdir, exist_ok=True)
        np.save(os.path.join(current_outdir, 'voxel_sparse.npy'), voxel_sparse.astype(np.int16))
        np.save(os.path.join(current_outdir, 'height_map.npy'), current_height_map.numpy())
        np.save(os.path.join(current_outdir, 'semantic_map.npy'), current_semantic_map.numpy())
        np.save(os.path.join(current_outdir, 'hmap_mc.npy'), heightmap.numpy())
