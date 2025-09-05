# Interpretable Auto Window Setting for Deep-Learning-Based CT Analysis

*Yiqin Zhang, Meiling Chen, Zhengjie Zhang*

🎉 2025.08.31: The paper has been accepted by CBM. 🎉
📄 [CBM paper link](https://www.sciencedirect.com/science/article/pii/S0010482525013460)
📚 [ArXiv](https://arxiv.org/abs/2501.06223)

## Pre-requisites

- Python 3.10+
- PyTorch 2.4.0+
- ~~mgamdata package: [Gitee Repo](https://gitee.com/MGAM/mgam_datatoolkit)~~
- ITKIT package: [Gitee Repo](https://gitee.com/MGAM/itkit)
- mmsegmentation redistribution: [Gitee Repo](https://gitee.com/MGAM/mmsegmentation)
- mmengine redistribution: [Gitee Repo](https://gitee.com/MGAM/mmengine)
- mmpretrain redistribution: [Gitee Repo](https://gitee.com/MGAM/mmpretrain)
- NVIDIA CUDA

This research heavily relies on our `ITKIT` package, the AutoWindow Module can be found in [Gitee Repo](`https://gitee.com/MGAM/itkit/blob/v1.11/mgamdata/models/AutoWindow.py`).

## Environment Setup

```bash
export mm_workdir="..." # runner workdir
export mm_testdir="..." # runner testdir
export mm_configdir="..." # runner configdir
export supported_models="SegFormer3D,MedNeXt" # supported models
```

## Data Preparation

### Source Data

```plain
/root/data
├── image
│   ├── 00001.nii
│   ├── 00002.nii
│   ├── 00003.nii
│   └── ...
│
└── label
    ├── 00001.nii
    ├── 00002.nii
    ├── 00003.nii
    └── ...
```

### Preprocess

#### Resample

```bash
itk_resample \
    /root/data \
    /root/data_resampled \
    --mp \
    --spacing 2 1 1 # ZYX (You may change to your own spacing)
```

#### Patch

```bash
split3d \
    /root/data_resampled \
    /root/data_patched \
    --window-size 16 \
    --stride 8 \
    --mp
```

### The desired data structure

```plain
/root
├── data
│   ├── image
│   │   ├── 00001.nii
│   │   ├── 00002.nii
│   │   ├── 00003.nii
│   │   └── ...
│   └── label
│       ├── 00001.nii
│       ├── 00002.nii
│       ├── 00003.nii
│       └── ...
│
├── data_resampled
│   ├── image
│   │   ├── 00001.mha
│   │   ├── 00002.mha
│   │   ├── 00003.mha
│   │   └── ...
│   └── label
│       ├── 00001.mha
│       ├── 00002.mha
│       ├── 00003.mha
│       └── ...
│
└── data_patched
    ├── 00001
    │   ├── 0.npz
    │   ├── 1.npz
    │   ├── 2.npz
    │   └── ...
    ├── 00002
    │   ├── 0.npz
    │   ├── 1.npz
    │   ├── 2.npz
    │   └── ...
    ├── 00003
    │   ├── 0.npz
    │   ├── 1.npz
    │   ├── 2.npz
    │   └── ...
    └── ...

```

## Configuration

You may have to modify the config file according to specify your dataset path. All configs are stored in `configs/` folder. If you do not want to run the implementation, the configs can provide sufficient information for you to understand the implementation.

## Run

```bash
mmrun {ConfigVersionPrefix}
```

ConfigVersionPrefix Example: `1.1.4.1`

## Email

For any questions, please contact us via email:

[Yiqin Zhang](mailto:312065559@qq.com) (Corresponding Author)

## Citations

```bibtex
@article{ZHANG2025110994,
    title = {Interpretable Auto Window setting for deep-learning-based CT analysis},
    journal = {Computers in Biology and Medicine},
    volume = {197},
    pages = {110994},
    year = {2025},
    issn = {0010-4825},
    doi = {https://doi.org/10.1016/j.compbiomed.2025.110994},
    url = {https://www.sciencedirect.com/science/article/pii/S0010482525013460},
    author = {Yiqin Zhang and Meiling Chen and Zhengjie Zhang},
    keywords = {Deep learning, Medical image analysis, Computed tomography, Multi-window processing, Medical fundamental models},
}
```
