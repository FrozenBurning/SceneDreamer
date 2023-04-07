import argparse

import os
import torch

from imaginaire.config import Config
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.dataset import get_test_dataloader
from imaginaire.utils.distributed import init_dist
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.utils.io import get_checkpoint as get_checkpoint
from imaginaire.utils.logging import init_logging
from imaginaire.utils.trainer import \
    (get_model_optimizer_and_scheduler, set_random_seed)
import imaginaire.config


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', required=True,
                        help='Path to the training config file.')
    parser.add_argument('--checkpoint', default='',
                        help='Checkpoint path.')
    parser.add_argument('--output_dir', required=True,
                        help='Location to save the image outputs')
    parser.add_argument('--logdir',
                        help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu_id', type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    set_affinity(args.local_rank)
    set_random_seed(args.seed, by_rank=True)
    cfg = Config(args.config)
    imaginaire.config.DEBUG = args.debug

    if not hasattr(cfg, 'inference_args'):
        cfg.inference_args = None

    # If args.single_gpu is set to True,
    # we will disable distributed data parallel.
    if not args.single_gpu:
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank)

    # Override the number of data loading workers if necessary
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    # Create log directory for storing training results.
    cfg.date_uid, cfg.logdir = init_logging(args.config, args.logdir)

    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    # Initialize data loaders and models.
    net_G = get_model_optimizer_and_scheduler(cfg, seed=args.seed, generator_only=True)

    if args.checkpoint == '':
        raise NotImplementedError("No checkpoint is provided for inference!")

    # Load checkpoint.
    # trainer.load_checkpoint(cfg, args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    net_G.load_state_dict(checkpoint['net_G'])

    # Do inference.
    net_G = net_G.module
    net_G.eval()
    for name, param in net_G.named_parameters():
        param.requires_grad = False
    torch.cuda.empty_cache()
    device = torch.device('cuda')
    rng_cuda = torch.Generator(device=device)
    rng_cuda = rng_cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    world_dir = os.path.join(args.output_dir)
    os.makedirs(world_dir, exist_ok=True)
    print('[PCGGenerator] Generating BEV scene representation...')
    os.system('python terrain_generator.py --size {} --seed {} --outdir {}'.format(net_G.voxel.sample_size, args.seed, world_dir))
    net_G.voxel.next_world(device, world_dir, checkpoint)
    cam_mode = cfg.inference_args.camera_mode
    current_outdir = os.path.join(world_dir, 'camera_{:02d}'.format(cam_mode))
    os.makedirs(current_outdir, exist_ok=True)
    os.makedirs(current_outdir, exist_ok=True)
    z = torch.empty(1, net_G.style_dims, dtype=torch.float32, device=device)
    z.normal_(generator=rng_cuda)
    net_G.inference_givenstyle(z, current_outdir, **vars(cfg.inference_args))

if __name__ == "__main__":
    main()
