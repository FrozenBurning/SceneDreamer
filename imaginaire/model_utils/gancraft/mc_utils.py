import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import csv
import time
from scipy import ndimage
import os

from imaginaire.model_utils.gancraft.mc_lbl_reduction import ReducedLabelMapper


def load_voxel_new(voxel_path, shape=[256, 512, 512]):
    voxel_world = np.fromfile(voxel_path, dtype='int32')
    voxel_world = voxel_world.reshape(
        shape[1]//16, shape[2]//16, 16, 16, shape[0])
    voxel_world = voxel_world.transpose(4, 0, 2, 1, 3)
    voxel_world = voxel_world.reshape(shape[0], shape[1], shape[2])
    voxel_world = np.ascontiguousarray(voxel_world)
    voxel_world = torch.from_numpy(voxel_world.astype(np.int32))
    return voxel_world


def gen_corner_voxel(voxel):
    r"""Converting voxel center array to voxel corner array. The size of the
    produced array grows by 1 on every dimension.

    Args:
        voxel (torch.IntTensor, CPU): Input voxel of three dimensions
    """
    structure = np.zeros([3, 3, 3], dtype=np.bool)
    structure[1:, 1:, 1:] = True
    voxel_p = F.pad(voxel, (0, 1, 0, 1, 0, 1))
    corners = ndimage.binary_dilation(voxel_p.numpy(), structure)
    corners = torch.tensor(corners, dtype=torch.int32)
    return corners


def calc_height_map(voxel_t):
    r"""Calculate height map given a voxel grid [Y, X, Z] as input.
    The height is defined as the Y index of the surface (non-air) block

    Args:
        voxel (Y x X x Z torch.IntTensor, CPU): Input voxel of three dimensions
    Output:
        heightmap (X x Z torch.IntTensor)
    """
    m, h = torch.max((torch.flip(voxel_t, [0]) != 0).int(), dim=0, keepdim=False)
    heightmap = voxel_t.shape[0] - 1 - h
    heightmap[m == 0] = 0  # Special case when the whole vertical column is empty
    return heightmap


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


def cumsum_exclusive(tensor, dim):
    cumsum = torch.cumsum(tensor, dim)
    cumsum = torch.roll(cumsum, 1, dim)
    cumsum.index_fill_(dim, torch.tensor([0], dtype=torch.long, device=tensor.device), 0)
    return cumsum


def sample_depth_batched(depth2, nsamples, deterministic=False, use_box_boundaries=True, sample_depth=4):
    r"""    Make best effort to sample points within the same distance for every ray.
    Exception: When there is not enough voxel.

    Args:
        depth2 (N x 2 x 256 x 256 x 4 x 1 tensor):
        - N: Batch.
        - 2: Entrance / exit depth for each intersected box.
        - 256, 256: Height, Width.
        - 4: Number of intersected boxes along the ray.
        - 1: One extra dim for consistent tensor dims.
        depth2 can include NaNs.
        deterministic (bool): Whether to use equal-distance sampling instead of random stratified sampling.
        use_box_boundaries (bool): Whether to add the entrance / exit points into the sample.
        sample_depth (float): Truncate the ray when it travels further than sample_depth inside voxels.
    """

    bs = depth2.size(0)
    dim0 = depth2.size(2)
    dim1 = depth2.size(3)
    dists = depth2[:, 1] - depth2[:, 0]
    dists[torch.isnan(dists)] = 0  # N, 256, 256, 4, 1
    accu_depth = torch.cumsum(dists, dim=-2)  # N, 256, 256, 4, 1
    total_depth = accu_depth[..., [-1], :]  # N, 256, 256, 1, 1

    total_depth = torch.clamp(total_depth, None, sample_depth)

    # Ignore out of range box boundaries. Fill with random samples.
    if use_box_boundaries:
        boundary_samples = accu_depth.clone().detach()
        boundary_samples_filler = torch.rand_like(boundary_samples) * total_depth
        bad_mask = (accu_depth > sample_depth) | (dists == 0)
        boundary_samples[bad_mask] = boundary_samples_filler[bad_mask]

    rand_shape = [bs, dim0, dim1, nsamples, 1]
    # 256, 256, N, 1
    if deterministic:
        rand_samples = torch.empty(rand_shape, dtype=total_depth.dtype, device=total_depth.device)
        rand_samples[..., :, 0] = torch.linspace(0, 1, nsamples+2)[1:-1]
    else:
        rand_samples = torch.rand(rand_shape, dtype=total_depth.dtype, device=total_depth.device)  # 256, 256, N, 1
        # Stratified sampling as in NeRF
        rand_samples = rand_samples / nsamples
        rand_samples[..., :, 0] += torch.linspace(0, 1, nsamples+1, device=rand_samples.device)[:-1]
    rand_samples = rand_samples * total_depth  # 256, 256, N, 1

    # Can also include boundaries
    if use_box_boundaries:
        rand_samples = torch.cat([rand_samples, boundary_samples, torch.zeros(
            [bs, dim0, dim1, 1, 1], dtype=total_depth.dtype, device=total_depth.device)], dim=-2)
    rand_samples, _ = torch.sort(rand_samples, dim=-2, descending=False)

    midpoints = (rand_samples[..., 1:, :] + rand_samples[..., :-1, :]) / 2
    new_dists = rand_samples[..., 1:, :] - rand_samples[..., :-1, :]

    # Scatter the random samples back
    # 256, 256, 1, M, 1 > 256, 256, N, 1, 1
    idx = torch.sum(midpoints.unsqueeze(-3) > accu_depth.unsqueeze(-2), dim=-3)  # 256, 256, M, 1
    # print(idx.shape, idx.max(), idx.min()) # max 3, min 0

    depth_deltas = depth2[:, 0, :, :, 1:, :] - depth2[:, 1, :, :, :-1, :]  # There might be NaNs!
    depth_deltas = torch.cumsum(depth_deltas, dim=-2)
    depth_deltas = torch.cat([depth2[:, 0, :, :, [0], :], depth_deltas+depth2[:, 0, :, :, [0], :]], dim=-2)
    heads = torch.gather(depth_deltas, -2, idx)  # 256 256 M 1
    # heads = torch.gather(depth2[0], -2, idx) # 256 256 M 1

    # print(torch.any(torch.isnan(heads)))
    rand_depth = heads + midpoints  # 256 256 N 1

    return rand_depth, new_dists, idx


def volum_rendering_relu(sigma, dists, dim=2):
    free_energy = F.relu(sigma) * dists

    a = 1 - torch.exp(-free_energy.float())  # probability of it is not empty here
    b = torch.exp(-cumsum_exclusive(free_energy, dim=dim))  # probability of everything is empty up to now
    probs = a * b  # probability of the ray hits something here

    return probs

class MCLabelTranslator:
    r"""Resolving mapping across Minecraft voxel, coco-stuff label and reduced label set."""

    def __init__(self):
        this_path = os.path.dirname(os.path.abspath(__file__))
        # Load voxel name lut
        id2name_lut = {}
        id2color_lut = {}
        id2glbl_lut = {}
        with open(os.path.join(this_path, 'id2name_gg.csv'), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                id2name_lut[int(row[0])] = row[1]
                id2color_lut[int(row[0])] = int(row[2])
                id2glbl_lut[int(row[0])] = row[3]

        # Load GauGAN color lut
        glbl2color_lut = {}
        glbl2cocoidx_lut = {}
        with open(os.path.join(this_path, 'gaugan_lbl2col.csv'), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            cocoidx = 1  # 0 is "Others"
            for row in csvreader:
                color = int(row[1].lstrip('#'), 16)
                glbl2color_lut[row[0]] = color
                glbl2cocoidx_lut[row[0]] = cocoidx
                cocoidx += 1

        # Generate id2ggcolor lut
        id2ggcolor_lut = {}
        for k, v in id2glbl_lut.items():
            if v:
                id2ggcolor_lut[k] = glbl2color_lut[v]
            else:
                id2ggcolor_lut[k] = 0

        # Generate id2cocoidx
        id2cocoidx_lut = {}
        for k, v in id2glbl_lut.items():
            if v:
                id2cocoidx_lut[k] = glbl2cocoidx_lut[v]
            else:
                id2cocoidx_lut[k] = 0

        self.id2color_lut = id2color_lut
        self.id2name_lut = id2name_lut
        self.id2glbl_lut = id2glbl_lut
        self.id2ggcolor_lut = id2ggcolor_lut
        self.id2cocoidx_lut = id2cocoidx_lut

        if True:
            mapper = ReducedLabelMapper()
            mcid2rdid_lut = mapper.mcid2rdid_lut
            mcid2rdid_lut = torch.tensor(mcid2rdid_lut, dtype=torch.long)
            self.mcid2rdid_lut = mcid2rdid_lut
            self.num_reduced_lbls = len(mapper.reduced_lbls)
            self.ignore_id = mapper.ignore_id
            self.dirt_id = mapper.dirt_id
            self.water_id = mapper.water_id

            self.mapper = mapper

            ggid2rdid_lut = mapper.ggid2rdid + [0]  # Last index is ignore
            ggid2rdid_lut = torch.tensor(ggid2rdid_lut, dtype=torch.long)
            self.ggid2rdid_lut = ggid2rdid_lut
        if True:
            mc2coco_lut = list(zip(*sorted([(k, v) for k, v in self.id2cocoidx_lut.items()])))[1]
            mc2coco_lut = torch.tensor(mc2coco_lut, dtype=torch.long)
            self.mc2coco_lut = mc2coco_lut

    def gglbl2ggid(self, gglbl):
        return self.mapper.gglbl2ggid[gglbl]

    def mc2coco(self, mc):
        self.mc2coco_lut = self.mc2coco_lut.to(mc.device)
        coco = self.mc2coco_lut[mc.long()]
        return coco

    def mc2reduced(self, mc, ign2dirt=False):
        self.mcid2rdid_lut = self.mcid2rdid_lut.to(mc.device)
        reduced = self.mcid2rdid_lut[mc.long()]
        if ign2dirt:
            reduced[reduced == self.ignore_id] = self.dirt_id
        return reduced

    def coco2reduced(self, coco):
        self.ggid2rdid_lut = self.ggid2rdid_lut.to(coco.device)
        reduced = self.ggid2rdid_lut[coco.long()]
        return reduced

    def get_num_reduced_lbls(self):
        return self.num_reduced_lbls

    @staticmethod
    def uint32_to_4uint8(x):
        dt1 = np.dtype(('i4', [('bytes', 'u1', 4)]))
        color = x.view(dtype=dt1)['bytes']
        return color

    def mc_color(self, img):
        r"""Obtaining Minecraft default color.

        Args:
            img (H x W x 1 int32 numpy tensor): Segmentation map.
        """
        lut = self.id2color_lut
        lut = list(zip(*sorted([(k, v) for k, v in lut.items()])))[1]
        lut = np.array(lut, dtype=np.uint32)
        rgb = lut[img]
        rgb = self.uint32_to_4uint8(rgb)[..., :3]

        return rgb


def rand_crop(cam_c, cam_res, target_res):
    r"""Produces a new cam_c so that the effect of rendering with the new cam_c and target_res is the same as rendering
    with the old parameters and then crop out target_res.
    """
    d0 = np.random.randint(cam_res[0] - target_res[0] + 1)
    d1 = np.random.randint(cam_res[1] - target_res[1] + 1)
    cam_c = [cam_c[0]-d0, cam_c[1]-d1]
    return cam_c


def segmask_smooth(seg_mask, kernel_size=7):
    labels = F.avg_pool2d(seg_mask, kernel_size, 1, kernel_size//2)
    onehot_idx = torch.argmax(labels, dim=1, keepdims=True)
    labels.fill_(0.0)
    labels.scatter_(1, onehot_idx, 1.0)
    return labels


def colormap(x, cmap='viridis'):
    x = np.nan_to_num(x, np.nan, np.nan, np.nan)
    x = x - np.nanmin(x)
    x = x / np.nanmax(x)
    rgb = plt.get_cmap(cmap)(x)[..., :3]
    return rgb
