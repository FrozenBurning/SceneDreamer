<div align="center">

<h1>SceneDreamer: Unbounded 3D Scene Generation from 2D Image Collections</h1>

<div>
    <a href='https://frozenburning.github.io/' target='_blank'>Zhaoxi Chen</a>&emsp;
    <a href='https://wanggcong.github.io/' target='_blank'>Guangcong Wang</a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a>
</div>
<div>
    S-Lab, Nanyang Technological University
</div>

<div>

<a target="_blank" href="https://arxiv.org/abs/2302.01330">
  <img src="https://img.shields.io/badge/arXiv-2302.01330-b31b1b.svg" alt="arXiv Paper"/>
</a>
<a target="_blank" href="https://huggingface.co/spaces/FrozenBurning/SceneDreamer">
  <img src="https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue" alt="HuggingFace"/>
</a>
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FFrozenBurning%2FSceneDreamer&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</div>


<h4>TL;DR</h4>
<h5>SceneDreamer learns to generate unbounded 3D scenes from in-the-wild 2D image collections. <br> Our method can synthesize diverse landscapes across different styles, with 3D consistency, well-defined depth, and free camera trajectory.</h5>

### [Paper](https://arxiv.org/abs/2302.01330) | [Project Page](https://scene-dreamer.github.io/) | [Video](https://youtu.be/nEfSKL2_FoA) | [Hugging Face :hugs:](https://huggingface.co/spaces/FrozenBurning/SceneDreamer)

<br>


<tr>
    <img src="./assets/teaser.gif" width="100%"/>
</tr>

</div>

## Updates

[04/2023] Hugging Face demo released! [![demo](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/FrozenBurning/SceneDreamer)

[04/2023] Inference code released!

[02/2023] Paper uploaded to arXiv. [![arXiv](https://img.shields.io/badge/arXiv-2302.01330-b31b1b.svg)](https://arxiv.org/abs/2302.01330)

## Citation
If you find our work useful for your research, please consider citing this paper:
```
@inproceedings{chen2023sd,
    title={SceneDreamer: Unbounded 3D Scene Generation from 2D Image Collections},
    author={Chen, Zhaoxi and Wang, Guangcong and Liu, Ziwei},
    booktitle={arXiv},
    year={2023},
}
```

## Installation
We highly recommend using [Anaconda](https://www.anaconda.com/) to manage your python environment. You can setup the required environment by the following commands:
```bash
# install python dependencies
conda env create -f environment.yml
conda activate scenedreamer

# compile third party libraries
export CUDA_VERSION=$(nvcc --version| grep -Po "(\d+\.)+\d+" | head -1)
CURRENT=$(pwd)
for p in correlation channelnorm resample2d bias_act upfirdn2d; do
    cd imaginaire/third_party/${p};
    rm -rf build dist *info;
    python setup.py install;
    cd ${CURRENT};
done

for p in gancraft/voxlib; do
  cd imaginaire/model_utils/${p};
  make all
  cd ${CURRENT};
done

cd gridencoder
python setup.py build_ext --inplace
python -m pip install .
cd ${CURRENT}

# Now, all done!
```

## Inference

### Download Pretrained Models
Please download our checkpoints from [Google Drive](https://drive.google.com/file/d/1IFu1vNrgF1EaRqPizyEgN_5Vt7Fyg0Mj/view?usp=share_link) to run the following inference scripts. You may store the checkpoint at the root directory of this repo:
```
├── ...
└── SceneDreamer
    ├── inference.py
    ├── README.md
    └── scenedreamer_released.pt
```

### Render!
You can run the following command to generate your own 3D world!
```bash
python inference.py --config configs/scenedreamer_inference.yaml --output_dir ./test/ --seed 8888 --checkpoint ./scenedreamer_released.pt
```

The results will be saved under `./test` as the following structures:
```
├── ...
└── test
    └── camera_{:02d} # camera mode for trajectory
        ├── rgb_render # per frame RGB renderings
            ├── 00000.png
            ├── 00001.png
            └── ...
        ├── rgb_render.mp4 # rendered video
        ├── height_map.png # height map
        ├── semantic_map.png # semantic map
        └── style.npy # sampled style code
```

Furthermore, you can modify the parameters for rendering in [scenedreamer_inference.yaml](./configs/scenedreamer_inference.yaml), detailed as follows:

| Parameter | Recommended Range | Description |
| :---------- | :------------: | :---------- |
| `cam_mode` | 0 - 9 | Different camera trajectries for rendered sequence |
| `cam_maxstep` | 0 - ∞ | Total number of frames. Increase for a more smooth camera movement. |
| `resolution_hw` | [540, 960] - [2160, 3840] | The resolution of each rendered frame |
| `num_samples` | 12 - 40 | The number of sampled points per camera ray |
| `cam_ang` | 50 - 130 | The FOV of camera view |
| `scene_size` | 1024 - 2048 | The spatial resolution of sampled scene. |

Here is a sampled scene with our default rendering parameters:
<tr>
    <img src="./assets/sample_traj.gif" width="100%"/>
</tr>


## License

Distributed under the S-Lab License. See [LICENSE](./LICENSE) for more information. Part of the codes are also subject to the [LICENSE of imaginaire](https://github.com/NVlabs/imaginaire/blob/master/LICENSE.md) by NVIDIA.

## Acknowledgements
This work is supported by the National Research Foundation, Singapore under its AI Singapore Programme, NTU NAP, MOE AcRF Tier 2 (T2EP20221-0033), and under the RIE2020 Industry Alignment Fund - Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

SceneDreamer is implemented on top of the [imaginaire](https://github.com/NVlabs/imaginaire). Thanks [torch-ngp](https://github.com/ashawkey/torch-ngp) for the pytorch CUDA implementation of neural hash grid.