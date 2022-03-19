# TensoRF
## [Project page](https://apchenstu.github.io/TensoRF/) |  [Paper](https://arxiv.org/abs/2203.09517)
This repository contains a pytorch implementation for the paper: [TensoRF: Tensorial Radiance Fields](https://arxiv.org/abs/2103.15595). Our work present a novel approach to model and reconstruct radiance fields, which achieves super
**fast** training process, **compact** memory footprint and **state-of-the-art** rendering quality.<br><br>


https://user-images.githubusercontent.com/16453770/158920837-3fafaa17-6ed9-4414-a0b1-a80dc9e10301.mp4
## Installation

#### Tested on Ubuntu 20.04 + Pytorch 1.10.1 

Install environment:
```
conda create -n TensoRF python=3.8
conda activate TensoRF
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg
```


## Dataset
* [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
* [Synthetic-NSVF](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip)
* [Tanks&Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)
* [Forward-facing](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)



## Quick Start
The training script is in `train.py`, to train a TensoRF:

```
python train.py --config configs/lego.txt
```


we provide a few examples in the configuration folder, please note:

 `dataset_name`, choices = ['blender', 'llff', 'nsvf', 'dtu','tankstemple'];

 `shadingMode`, choices = ['MLP_Fea', 'SH'];

 `model_name`, choices = ['TensorVMSplit', 'TensorCP'], corresponding to the VM and CP decomposition;

 `n_lamb_sigma` and `n_lamb_sh` are string type refer to the basis number of density and appearance along XYZ
dimension;

 `N_voxel_init` and `N_voxel_final` control the resolution of matrix and vector;

 `N_vis` and `vis_every` control the visualization during training;


  You need to set `--render_test 1`/`--render_path 1` if you want to render testing views or path after training. 

More options refer to the `opt.py`. 

### For pretrained checkpoints and results please see:
[https://1drv.ms/u/s!Ard0t_p4QWIMgQ2qSEAs7MUk8hVw?e=dc6hBm](https://1drv.ms/u/s!Ard0t_p4QWIMgQ2qSEAs7MUk8hVw?e=dc6hBm),



## Rendering

```
python train.py --config configs/lego.txt --ckpt path/to/your/checkpoint --render_only 1 --render_test 1
```

You can just simply pass `--render_only 1` and `--ckpt path/to/your/checkpoint` to render images from a pre-trained
checkpoint. You may also need to specify what you want to render, like `--render_test 1`, `--render_train 1` or `--render_path 1`.
The rendering results are located in your checkpoint folder.

## Citation
If you find our code or paper helps, please consider citing:
```
@misc{TensoRF,
      title={TensoRF: Tensorial Radiance Fields},
      author={Anpei Chen and Zexiang Xu and Andreas Geiger and and Jingyi Yu and Hao Su},
      year={2022},
      eprint={2203.09517},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
