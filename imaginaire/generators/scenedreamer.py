# Using Hashgrid as backbone representation

import os
import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import imaginaire.model_utils.gancraft.camctl as camctl
import imaginaire.model_utils.gancraft.mc_utils as mc_utils
import imaginaire.model_utils.gancraft.voxlib as voxlib
from imaginaire.model_utils.pcg_gen import PCGVoxelGenerator
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.generators.gancraft_base import Base3DGenerator
from encoding import get_encoder

from imaginaire.model_utils.layers import LightningMLP, ConditionalHashGrid

class Generator(Base3DGenerator):
    r"""SceneDreamer generator constructor.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, gen_cfg, data_cfg):
        super(Generator, self).__init__(gen_cfg, data_cfg)
        print('SceneDreamer[Hash] on ALL Scenes generator initialization.')

        # here should be a list of height maps and semantic maps
        if gen_cfg.pcg_cache:
            raise NotImplementedError('gen_cfg.pcg_cache={} not supported!'.format(gen_cfg.pcg_cache))
        else:
            self.voxel = PCGVoxelGenerator(gen_cfg.scene_size)
        self.blk_feats = None
        # Minecraft -> SPADE label translator.
        self.label_trans = mc_utils.MCLabelTranslator()
        self.num_reduced_labels = self.label_trans.get_num_reduced_lbls()
        self.reduced_label_set = getattr(gen_cfg, 'reduced_label_set', False)
        self.use_label_smooth = getattr(gen_cfg, 'use_label_smooth', False)
        self.use_label_smooth_real = getattr(gen_cfg, 'use_label_smooth_real', self.use_label_smooth)
        self.use_label_smooth_pgt = getattr(gen_cfg, 'use_label_smooth_pgt', False)
        self.label_smooth_dia = getattr(gen_cfg, 'label_smooth_dia', 11)

        # Load MLP model.
        self.hash_encoder, self.hash_in_dim = get_encoder(encoding='hashgrid', input_dim=5, desired_resolution=2048 * 1, level_dim=8)
        self.render_net = LightningMLP(self.hash_in_dim, viewdir_dim=self.input_dim_viewdir, style_dim=self.interm_style_dims, mask_dim=self.num_reduced_labels, out_channels_s=1, out_channels_c=self.final_feat_dim, **self.mlp_model_kwargs)
        print(self.hash_encoder)
        self.world_encoder = ConditionalHashGrid()

        print('Done with the SceneDreamer initialization.')

    def custom_init(self):
        r"""Weight initialization."""

        def init_func(m):
            if hasattr(m, 'weight'):
                try:
                    nn.init.kaiming_normal_(m.weight.data, a=0.2, nonlinearity='leaky_relu')
                except:
                    print(m.name)
                m.weight.data *= 0.5
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.fill_(0.0)
        self.apply(init_func)

    def _forward_perpix_sub(self, blk_feats, worldcoord2, raydirs_in, z, mc_masks_onehot=None, global_enc=None):
        r"""Per-pixel rendering forwarding

        Args:
            blk_feats: Deprecated
            worldcoord2 (N x H x W x L x 3 tensor): 3D world coordinates of sampled points. L is number of samples; N is batch size, always 1.
            raydirs_in (N x H x W x 1 x C2 tensor or None): ray direction embeddings.
            z (N x C3 tensor): Intermediate style vectors.
            mc_masks_onehot (N x H x W x L x C4): One-hot segmentation maps.
        Returns:
            net_out_s (N x H x W x L x 1 tensor): Opacities.
            net_out_c (N x H x W x L x C5 tensor): Color embeddings.
        """
        _x, _y, _z = self.voxel.voxel_t.shape
        delimeter = torch.Tensor([_x, _y, _z]).to(worldcoord2)
        normalized_coord = worldcoord2 / delimeter * 2 - 1
        global_enc = global_enc[:, None, None, None, :].repeat(1, normalized_coord.shape[1], normalized_coord.shape[2], normalized_coord.shape[3], 1)
        normalized_coord = torch.cat([normalized_coord, global_enc], dim=-1)
        feature_in = self.hash_encoder(normalized_coord)

        net_out_s, net_out_c = self.render_net(feature_in, raydirs_in, z, mc_masks_onehot)

        if self.raw_noise_std > 0.:
            noise = torch.randn_like(net_out_s) * self.raw_noise_std
            net_out_s = net_out_s + noise

        return net_out_s, net_out_c

    def _forward_perpix(self, blk_feats, voxel_id, depth2, raydirs, cam_ori_t, z, global_enc):
        r"""Sample points along rays, forwarding the per-point MLP and aggregate pixel features

        Args:
            blk_feats (K x C1 tensor): Deprecated
            voxel_id (N x H x W x M x 1 tensor): Voxel ids from ray-voxel intersection test. M: num intersected voxels, why always 6?
            depth2 (N x 2 x H x W x M x 1 tensor): Depths of entrance and exit points for each ray-voxel intersection.
            raydirs (N x H x W x 1 x 3 tensor): The direction of each ray.
            cam_ori_t (N x 3 tensor): Camera origins.
            z (N x C3 tensor): Intermediate style vectors.
        """
        # Generate sky_mask; PE transform on ray direction.
        with torch.no_grad():
            raydirs_in = raydirs.expand(-1, -1, -1, 1, -1).contiguous()
            if self.pe_params[2] == 0 and self.pe_params[3] is True:
                raydirs_in = raydirs_in
            elif self.pe_params[2] == 0 and self.pe_params[3] is False:  # Not using raydir at all
                raydirs_in = None
            else:
                raydirs_in = voxlib.positional_encoding(raydirs_in, self.pe_params[2], -1, self.pe_params[3])

            # sky_mask: when True, ray finally hits sky
            sky_mask = voxel_id[:, :, :, [-1], :] == 0
            # sky_only_mask: when True, ray hits nothing but sky
            sky_only_mask = voxel_id[:, :, :, [0], :] == 0

        with torch.no_grad():
            # Random sample points along the ray
            num_samples = self.num_samples + 1
            if self.sample_use_box_boundaries:
                num_samples = self.num_samples - self.num_blocks_early_stop

            # 10 samples per ray + 4 intersections - 2
            rand_depth, new_dists, new_idx = mc_utils.sample_depth_batched(
                depth2, num_samples, deterministic=self.coarse_deterministic_sampling,
                use_box_boundaries=self.sample_use_box_boundaries, sample_depth=self.sample_depth)

            nan_mask = torch.isnan(rand_depth)
            inf_mask = torch.isinf(rand_depth)
            rand_depth[nan_mask | inf_mask] = 0.0

            worldcoord2 = raydirs * rand_depth + cam_ori_t[:, None, None, None, :]

            # Generate per-sample segmentation label
            voxel_id_reduced = self.label_trans.mc2reduced(voxel_id, ign2dirt=True)
            mc_masks = torch.gather(voxel_id_reduced, -2, new_idx)  # B 256 256 N 1
            mc_masks = mc_masks.long()
            mc_masks_onehot = torch.zeros([mc_masks.size(0), mc_masks.size(1), mc_masks.size(
                2), mc_masks.size(3), self.num_reduced_labels], dtype=torch.float, device=voxel_id.device)
            # mc_masks_onehot: [B H W Nlayer 680]
            mc_masks_onehot.scatter_(-1, mc_masks, 1.0)

        net_out_s, net_out_c = self._forward_perpix_sub(blk_feats, worldcoord2, raydirs_in, z, mc_masks_onehot, global_enc)

        # Handle sky
        sky_raydirs_in = raydirs.expand(-1, -1, -1, 1, -1).contiguous()
        sky_raydirs_in = voxlib.positional_encoding(sky_raydirs_in, self.pe_params_sky[0], -1, self.pe_params_sky[1])
        skynet_out_c = self.sky_net(sky_raydirs_in, z)

        # Blending
        weights = mc_utils.volum_rendering_relu(net_out_s, new_dists * self.dists_scale, dim=-2)

        # If a ray exclusively hits the sky (no intersection with the voxels), set its weight to zero.
        weights = weights * torch.logical_not(sky_only_mask).float()
        total_weights_raw = torch.sum(weights, dim=-2, keepdim=True)  # 256 256 1 1
        total_weights = total_weights_raw

        is_gnd = worldcoord2[..., [0]] <= 1.0  # Y X Z, [256, 256, 4, 3], nan < 1.0 == False
        is_gnd = is_gnd.any(dim=-2, keepdim=True)
        nosky_mask = torch.logical_or(torch.logical_not(sky_mask), is_gnd)
        nosky_mask = nosky_mask.float()

        # Avoid sky leakage
        sky_weight = 1.0-total_weights
        if self.keep_sky_out:
            # keep_sky_out_avgpool overrides sky_replace_color
            if self.sky_replace_color is None or self.keep_sky_out_avgpool:
                if self.keep_sky_out_avgpool:
                    if hasattr(self, 'sky_avg'):
                        sky_avg = self.sky_avg
                    else:
                        if self.sky_global_avgpool:
                            sky_avg = torch.mean(skynet_out_c, dim=[1, 2], keepdim=True)
                        else:
                            skynet_out_c_nchw = skynet_out_c.permute(0, 4, 1, 2, 3).squeeze(-1).contiguous()
                            sky_avg = F.avg_pool2d(skynet_out_c_nchw, 31, stride=1, padding=15, count_include_pad=False)
                            sky_avg = sky_avg.permute(0, 2, 3, 1).unsqueeze(-2).contiguous()
                    # print(sky_avg.shape)
                    skynet_out_c = skynet_out_c * (1.0-nosky_mask) + sky_avg*(nosky_mask)
                else:
                    sky_weight = sky_weight * (1.0-nosky_mask)
            else:
                skynet_out_c = skynet_out_c * (1.0-nosky_mask) + self.sky_replace_color*(nosky_mask)

        if self.clip_feat_map is True:  # intermediate feature before blending & CNN
            rgbs = torch.clamp(net_out_c, -1, 1) + 1
            rgbs_sky = torch.clamp(skynet_out_c, -1, 1) + 1
            net_out = torch.sum(weights*rgbs, dim=-2, keepdim=True) + sky_weight * \
                rgbs_sky  # 576, 768, 4, 3 -> 576, 768, 3
            net_out = net_out.squeeze(-2)
            net_out = net_out - 1
        elif self.clip_feat_map is False:
            rgbs = net_out_c
            rgbs_sky = skynet_out_c
            net_out = torch.sum(weights*rgbs, dim=-2, keepdim=True) + sky_weight * \
                rgbs_sky  # 576, 768, 4, 3 -> 576, 768, 3
            net_out = net_out.squeeze(-2)
        elif self.clip_feat_map == 'tanh':
            rgbs = torch.tanh(net_out_c)
            rgbs_sky = torch.tanh(skynet_out_c)
            net_out = torch.sum(weights*rgbs, dim=-2, keepdim=True) + sky_weight * \
                rgbs_sky  # 576, 768, 4, 3 -> 576, 768, 3
            net_out = net_out.squeeze(-2)
        else:
            raise NotImplementedError

        return net_out, new_dists, weights, total_weights_raw, rand_depth, net_out_s, net_out_c, skynet_out_c, \
            nosky_mask, sky_mask, sky_only_mask, new_idx

    def forward(self, data, random_style=False):
        r"""SceneDreamer forward.
        """
        raise NotImplementedError()


    def inference_givenstyle(self, style,
                  output_dir,
                  camera_mode,
                  style_img_path=None,
                  seed=1,
                  pad=30,
                  num_samples=40,
                  num_blocks_early_stop=6,
                  sample_depth=3,
                  tile_size=128,
                  resolution_hw=[540, 960],
                  cam_ang=72,
                  cam_maxstep=10):
        r"""Compute result images according to the provided camera trajectory and save the results in the specified
        folder. The full image is evaluated in multiple tiles to save memory.

        Args:
            output_dir (str): Where should the results be stored.
            camera_mode (int): Which camera trajectory to use.
            style_img_path (str): Path to the style-conditioning image.
            seed (int): Random seed (controls style when style_image_path is not specified).
            pad (int): Pixels to remove from the image tiles before stitching. Should be equal or larger than the
            receptive field of the CNN to avoid border artifact.
            num_samples (int): Number of samples per ray (different from training).
            num_blocks_early_stop (int): Max number of intersected boxes per ray before stopping
            (different from training).
            sample_depth (float): Max distance traveled through boxes before stopping (different from training).
            tile_size (int): Max size of a tile in pixels.
            resolution_hw (list [H, W]): Resolution of the output image.
            cam_ang (float): Horizontal FOV of the camera (may be adjusted by the camera controller).
            cam_maxstep (int): Number of frames sampled from the camera trajectory.
        """

        def write_img(path, img, rgb_input=False):
            img = ((img*0.5+0.5)*255).detach().cpu().numpy().astype(np.uint8)
            img = img[0].transpose(1, 2, 0)
            if rgb_input:
                img = img[..., [2, 1, 0]]
            cv2.imwrite(path, img,  [cv2.IMWRITE_PNG_COMPRESSION, 4])
            return img[..., ::-1]

        def read_img(path):
            img = cv2.imread(path).astype(np.float32)[..., [2, 1, 0]].transpose(2, 0, 1) / 255
            img = img * 2 - 1
            img = torch.from_numpy(img)

        print('Saving to', output_dir)

        # Use provided random seed.
        device = torch.device('cuda')

        global_enc = self.world_encoder(self.voxel.current_height_map, self.voxel.current_semantic_map)

        biome_colors = torch.Tensor([
        [255, 255, 178],
        [184, 200, 98],
        [188, 161, 53],
        [190, 255, 242],
        [106, 144, 38],
        [33, 77, 41],
        [86, 179, 106],
        [34, 61, 53],
        [35, 114, 94],
        [0, 0, 255],
        [0, 255, 0],
        ]).to(device) / 255 * 2 - 1
        semantic_map = torch.argmax(self.voxel.current_semantic_map, dim=1)

        self.pad = pad
        self.num_samples = num_samples
        self.num_blocks_early_stop = num_blocks_early_stop
        self.sample_depth = sample_depth

        self.coarse_deterministic_sampling = True
        self.crop_size = resolution_hw
        self.cam_res = [self.crop_size[0]+self.pad, self.crop_size[1]+self.pad]
        self.use_label_smooth_pgt = False

        # Make output dirs.
        output_dir = os.path.join(output_dir, 'rgb_render')
        os.makedirs(output_dir, exist_ok=True)
        fout = imageio.get_writer(output_dir + '.mp4', fps=10)

        write_img(os.path.join(output_dir, 'semantic_map.png'), biome_colors[semantic_map].permute(0, 3, 1, 2), rgb_input=True)
        write_img(os.path.join(output_dir, 'height_map.png'), self.voxel.current_height_map)
        np.save(os.path.join(output_dir, 'style.npy'), style.detach().cpu().numpy())
        evalcamctl = camctl.EvalCameraController(
            self.voxel, maxstep=cam_maxstep, pattern=camera_mode, cam_ang=cam_ang,
            smooth_decay_multiplier=150/cam_maxstep)

        # Get output style.
        z = self.style_net(style)

        # Generate required output images.
        for id, (cam_ori_t, cam_dir_t, cam_up_t, cam_f) in enumerate(evalcamctl):
            print('Rendering frame', id)
            cam_f = cam_f * (self.crop_size[1]-1)  # So that the view is not depending on the padding
            cam_c = [(self.cam_res[0]-1)/2, (self.cam_res[1]-1)/2]

            voxel_id, depth2, raydirs = voxlib.ray_voxel_intersection_perspective(
                self.voxel.voxel_t, cam_ori_t, cam_dir_t, cam_up_t, cam_f, cam_c, self.cam_res,
                self.num_blocks_early_stop)

            voxel_id = voxel_id.unsqueeze(0)
            depth2 = depth2.unsqueeze(0)
            raydirs = raydirs.unsqueeze(0)
            cam_ori_t = cam_ori_t.unsqueeze(0).to(device)

            voxel_id_all = voxel_id
            depth2_all = depth2
            raydirs_all = raydirs

            # Evaluate sky in advance to get a consistent sky in the semi-transparent region.
            if self.sky_global_avgpool:
                sky_raydirs_in = raydirs.expand(-1, -1, -1, 1, -1).contiguous()
                sky_raydirs_in = voxlib.positional_encoding(
                    sky_raydirs_in, self.pe_params_sky[0], -1, self.pe_params_sky[1])
                skynet_out_c = self.sky_net(sky_raydirs_in, z)
                sky_avg = torch.mean(skynet_out_c, dim=[1, 2], keepdim=True)
                self.sky_avg = sky_avg

            num_strips_h = (self.cam_res[0]-self.pad+tile_size-1)//tile_size
            num_strips_w = (self.cam_res[1]-self.pad+tile_size-1)//tile_size

            fake_images_chunks_v = []
            # For each horizontal strip.
            for strip_id_h in range(num_strips_h):
                strip_begin_h = strip_id_h * tile_size
                strip_end_h = np.minimum(strip_id_h * tile_size + tile_size + self.pad, self.cam_res[0])
                # For each vertical strip.
                fake_images_chunks_h = []
                for strip_id_w in range(num_strips_w):
                    strip_begin_w = strip_id_w * tile_size
                    strip_end_w = np.minimum(strip_id_w * tile_size + tile_size + self.pad, self.cam_res[1])

                    voxel_id = voxel_id_all[:, strip_begin_h:strip_end_h, strip_begin_w:strip_end_w, :, :]
                    depth2 = depth2_all[:, :, strip_begin_h:strip_end_h, strip_begin_w:strip_end_w, :, :]
                    raydirs = raydirs_all[:, strip_begin_h:strip_end_h, strip_begin_w:strip_end_w, :, :]

                    net_out, new_dists, weights, total_weights_raw, rand_depth, net_out_s, net_out_c, skynet_out_c, \
                        nosky_mask, sky_mask, sky_only_mask, new_idx = self._forward_perpix(
                            self.blk_feats, voxel_id, depth2, raydirs, cam_ori_t, z, global_enc)
                    fake_images, _ = self._forward_global(net_out, z)

                    if self.pad != 0:
                        fake_images = fake_images[:, :, self.pad//2:-self.pad//2, self.pad//2:-self.pad//2]
                    fake_images_chunks_h.append(fake_images)
                fake_images_h = torch.cat(fake_images_chunks_h, dim=-1)
                fake_images_chunks_v.append(fake_images_h)
            fake_images = torch.cat(fake_images_chunks_v, dim=-2)
            rgb = write_img(os.path.join(output_dir,
                            '{:05d}.png'.format(id)), fake_images, rgb_input=True)
            fout.append_data(rgb)
        fout.close()
