# Interpretable Auto Window Setting for Deep-Learning-Based CT Analysis

Official Implementations
Preview V1.0

## Pre-requisites

- Python 3.10+
- PyTorch 2.4.0+
- mgamdata package: [Gitee Repo](https://gitee.com/MGAM/mgam_datatoolkit)
- mmsegmentation redistribution: [Gitee Repo](https://gitee.com/MGAM/mmsegmentation)
- mmengine redistribution: [Gitee Repo](https://gitee.com/MGAM/mmengine)
- mmpretrain redistribution: [Gitee Repo](https://gitee.com/MGAM/mmpretrain)
- NVIDIA CUDA

This research heavily relies on our mgamdata package, the AutoWindow Module can be found in [Gitee Repo](`https://gitee.com/MGAM/mgam_datatoolkit/blob/v1.11/mgamdata/models/AutoWindow.py`).

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

## Email

For any questions, please contact us via email:

[Yiqin Zhang](mailto:312065559@qq.com) (Corresponding Author)
