import os
import torch
import torch.nn as nn
import importlib
import argparse
from imaginaire.config import Config
from imaginaire.utils.cudnn import init_cudnn
import gradio as gr
from PIL import Image


class WrappedModel(nn.Module):
    r"""Dummy wrapping the module.
    """

    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        r"""PyTorch module forward function overload."""
        return self.module(*args, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', type=str, default='./configs/scenedreamer_inference.yaml', help='Path to the training config file.')
    parser.add_argument('--checkpoint', default='./scenedreamer_released.pt',
                        help='Checkpoint path.')
    parser.add_argument('--output_dir', type=str, default='./test/',
                        help='Location to save the image outputs')
    parser.add_argument('--seed', type=int, default=8888,
                        help='Random seed.')
    args = parser.parse_args()
    return args


args = parse_args()
cfg = Config(args.config)

# Initialize cudnn.
init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

# Initialize data loaders and models.

lib_G = importlib.import_module(cfg.gen.type)
net_G = lib_G.Generator(cfg.gen, cfg.data)
net_G = net_G.to('cuda')
net_G = WrappedModel(net_G)

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
world_dir = os.path.join(args.output_dir)
os.makedirs(world_dir, exist_ok=True)



def get_bev(seed):
    print('[PCGGenerator] Generating BEV scene representation...')
    os.system('python terrain_generator.py --size {} --seed {} --outdir {}'.format(net_G.voxel.sample_size, seed, world_dir))
    heightmap_path = os.path.join(world_dir, 'heightmap.png')
    semantic_path = os.path.join(world_dir, 'colormap.png')
    heightmap = Image.open(heightmap_path)
    semantic = Image.open(semantic_path)
    return semantic, heightmap

def get_video(seed, num_frames, reso_h, reso_w):
    device = torch.device('cuda')
    rng_cuda = torch.Generator(device=device)
    rng_cuda = rng_cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    net_G.voxel.next_world(device, world_dir, checkpoint)
    cam_mode = cfg.inference_args.camera_mode
    cfg.inference_args.cam_maxstep = num_frames
    cfg.inference_args.resolution_hw = [reso_h, reso_w]
    current_outdir = os.path.join(world_dir, 'camera_{:02d}'.format(cam_mode))
    os.makedirs(current_outdir, exist_ok=True)
    z = torch.empty(1, net_G.style_dims, dtype=torch.float32, device=device)
    z.normal_(generator=rng_cuda)
    net_G.inference_givenstyle(z, current_outdir, **vars(cfg.inference_args))
    return os.path.join(current_outdir, 'rgb_render.mp4')

markdown=f'''
  # SceneDreamer: Unbounded 3D Scene Generation from 2D Image Collections
  
  Authored by Zhaoxi Chen, Guangcong Wang, Ziwei Liu
  ### Useful links:
  - [Official Github Repo](https://github.com/FrozenBurning/SceneDreamer)
  - [Project Page](https://scene-dreamer.github.io/)
  - [arXiv Link](https://arxiv.org/abs/2302.01330)
  Licensed under the S-Lab License.
  We offer a sampled scene whose BEVs are shown on the right. You can also use the button "Generate BEV" to randomly sample a new 3D world represented by a height map and a semantic map. But it requires a long time. 
  
  To render video, push the button "Render" to generate a camera trajectory flying through the world. You can specify rendering options as shown below!
'''

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown(markdown)
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    semantic = gr.Image(value='./test/colormap.png',type="pil", shape=(512, 512))
                with gr.Column():
                    height = gr.Image(value='./test/heightmap.png', type="pil", shape=(512, 512))
            with gr.Row():
                # with gr.Column():
                #     image = gr.Image(type='pil', shape(540, 960))
                with gr.Column():
                    video = gr.Video()
    with gr.Row():
        num_frames = gr.Slider(minimum=10, maximum=200, value=20, step=1, label='Number of rendered frames')
        user_seed = gr.Slider(minimum=0, maximum=999999, value=8888, step=1, label='Random seed')
        resolution_h = gr.Slider(minimum=256, maximum=2160, value=270, step=1, label='Height of rendered image')
        resolution_w = gr.Slider(minimum=256, maximum=3840, value=480, step=1, label='Width of rendered image')

    with gr.Row():
        btn = gr.Button(value="Generate BEV")
        btn_2=gr.Button(value="Render")

    btn.click(get_bev,[user_seed],[semantic, height])
    btn_2.click(get_video,[user_seed, num_frames, resolution_h, resolution_w], [video])

demo.launch(debug=True)