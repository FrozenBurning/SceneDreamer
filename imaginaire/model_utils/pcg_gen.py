import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import cv2
import os

class PCGVoxelGenerator(nn.Module):
    def __init__(self, sample_size = 2048):
        super(PCGVoxelGenerator, self).__init__()
        self.sample_height = 256
        self.sample_size = sample_size
        self.voxel_t = None

    def next_world(self, device, world_dir, pcg_asset):
        # Generate BEV representation
        print('[PCGGenerator] Loading BEV scene representation...')
        heightmap_path = os.path.join(world_dir, 'heightmap.npy')
        semanticmap_path = os.path.join(world_dir, 'semanticmap.png')
        treemap_path = os.path.join(world_dir, 'treemap.png')
        height_map = np.load(heightmap_path)
        semantic_map = cv2.imread(semanticmap_path, 0)
        tree_map = cv2.imread(treemap_path, 0)

        print('[PCGGenerator] Creating scene windows...')
        height_map[height_map < 0] = 0
        height_map = ((height_map - height_map.min()) / (1 - height_map.min()) * (self.sample_height - 1)).astype(np.int16)

        self.total_size = height_map.shape


        org_semantic_map = torch.from_numpy(semantic_map.copy())
        org_semantic_map[tree_map != 255] = 10
        chunk_trees_map = tree_map

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
        chunk_grid_x, chunk_grid_y = torch.meshgrid(torch.arange(self.total_size[0]), torch.arange(self.total_size[1]))
        world_voxel_t = torch.zeros(self.sample_height, self.total_size[0], self.total_size[1]).to(torch.int32)

        chunk_height_map = torch.from_numpy(height_map.astype(int))[None, ...]
        chunk_semantic_map = torch.from_numpy(semantic_map)
        chunk_semantic_map = biome2mclabels[chunk_semantic_map[None, ...].long().contiguous()]
        world_voxel_t = world_voxel_t.scatter_(0, chunk_height_map, chunk_semantic_map)
        for preproc_step in range(16):
            world_voxel_t = world_voxel_t.scatter(0, torch.clip(chunk_height_map + preproc_step + 1, 0, self.sample_height - 1), chunk_semantic_map)
        chunk_height_map = chunk_height_map + 16
        chunk_height_map = chunk_height_map[0]
        boundary_detect = 50

        trees_models = pcg_asset['assets']

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
                if tree_pos_x[idx] < boundary_detect or tree_pos_x[idx] > self.total_size[0] - boundary_detect or tree_pos_y[idx] < boundary_detect or tree_pos_y[idx] > self.total_size[1] - boundary_detect or tree_pos_h[idx] > self.sample_height - boundary_detect:
                    # hack, to avoid out of index near the boundary
                    continue
                tree_id = random.choice(selected_trees)
                tmp = world_voxel_t[tree_pos_h[idx]: tree_pos_h[idx] + trees_models[tree_id].shape[0], tree_pos_x[idx]: tree_pos_x[idx] + trees_models[tree_id].shape[1], tree_pos_y[idx]: tree_pos_y[idx] + trees_models[tree_id].shape[2]]
                tmp_mask = (tmp == 0)
                try:
                    world_voxel_t[tree_pos_h[idx]: tree_pos_h[idx] + trees_models[tree_id].shape[0], tree_pos_x[idx]: tree_pos_x[idx] + trees_models[tree_id].shape[1], tree_pos_y[idx]: tree_pos_y[idx] + trees_models[tree_id].shape[2]][tmp_mask] = trees_models[tree_id][tmp_mask]
                except:
                    print('height?', tree_pos_h[idx])
                    print(tmp_mask.shape)
                    print(tmp.shape)
                    print(trees_models[tree_id].shape)
                    print(world_voxel_t.shape)
                    print(tree_id)
                    raise NotImplementedError
        self.trans_mat = torch.eye(4)  # Transform voxel to world
        # Generate heightmap for camera trajectory generation
        m, h = torch.max((torch.flip(world_voxel_t, [0]) != 0).int(), dim=0, keepdim=False)
        heightmap = world_voxel_t.shape[0] - 1 - h
        heightmap[m == 0] = 0  # Special case when the whole vertical column is empty
        gnd_level = heightmap.min()
        sky_level = heightmap.max() + 1
        current_height_map = (chunk_height_map / (self.sample_height - 1))[None, None, ...]
        current_semantic_map = F.one_hot(org_semantic_map.to(torch.int64)).to(torch.float).permute(2, 0, 1)[None, ...]

        self.current_height_map = current_height_map.to(device)
        self.current_semantic_map = current_semantic_map.to(device)
        self.heightmap = heightmap
        self.voxel_t = world_voxel_t[gnd_level:sky_level, :, :].to(device)
        self.trans_mat[0, 3] += gnd_level

    def world2local(self, v, is_vec=False):
        mat_world2local = torch.inverse(self.trans_mat)
        return trans_vec_homo(mat_world2local, v, is_vec)

    def is_sea(self, loc):
        r"""loc: [2]: x, z."""
        x = int(loc[1])
        z = int(loc[2])
        if x < 0 or x > self.heightmap.size(0) or z < 0 or z > self.heightmap.size(1):
            print('[McVoxel] is_sea(): Index out of bound.')
            return True
        y = self.heightmap[x, z] - self.trans_mat[0, 3]
        y = int(y)
        if self.voxel_t[y, x, z] == 26:
            print('[McVoxel] is_sea(): Get a sea.')
            print(self.voxel_t[y, x, z], self.voxel_t[y+1, x, z])
            return True
        else:
            return False
        
def trans_vec_homo(m, v, is_vec=False):
    r"""3-dimensional Homogeneous matrix and regular vector multiplication
    Convert v to homogeneous vector, perform M-V multiplication, and convert back
    Note that this function does not support autograd.

    Args:
        m (4 x 4 tensor): a homogeneous matrix
        v (3 tensor): a 3-d vector
        vec (bool): if true, v is direction. Otherwise v is point
    """
    if is_vec:
        v = torch.tensor([v[0], v[1], v[2], 0], dtype=v.dtype)
    else:
        v = torch.tensor([v[0], v[1], v[2], 1], dtype=v.dtype)
    v = torch.mv(m, v)
    if not is_vec:
        v = v / v[3]
    v = v[:3]
    return v