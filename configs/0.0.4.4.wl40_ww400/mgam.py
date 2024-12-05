from mmengine.config import read_base
with read_base():
    from ..base_totalsegmentator_dataset import *

from torch.optim.adamw import AdamW

# mmengine
from mmengine.runner import ValLoop
from mmengine.runner import TestLoop
from mmengine.hooks.iter_timer_hook import IterTimerHook
from mmengine.hooks.param_scheduler_hook import ParamSchedulerHook
from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.hooks import DistSamplerSeedHook
from mmengine.runner import IterBasedTrainLoop
from mmengine.optim.scheduler import LinearLR, PolyLR
from mmengine.optim import OptimWrapper, AmpOptimWrapper
from mmengine._strategy import FSDPStrategy
from mmengine.dataset.sampler import DefaultSampler, InfiniteSampler
from mmengine.visualization import LocalVisBackend
from mmengine.visualization import TensorboardVisBackend

# customize
from mgamdata.mm.mmeng_PlugIn import LoggerJSON
from mgamdata.mm.mmseg_PlugIn import IoUMetric_PerClass
from mgamdata.mm.mmeng_PlugIn import RemasteredDDP, RemasteredFSDP, RatioSampler
from mgamdata.process.GeneralPreProcess import WindowSet, TypeConvert, InstanceNorm
from mgamdata.process.LoadBiomedicalData import LoadImageFromMHA, LoadMaskFromMHA, LoadSampleFromNpz
from mgamdata.dataset.CT_ORG.mm_dataset import (CT_ORG_Mha, CT_ORG_Precrop_Npz)
from mgamdata.dataset.base import ParseID
from mgamdata.mm.mmseg_Dev3D import Seg3DDataPreProcessor, Seg3DLocalVisualizer, Seg3DVisualizationHook
from mgamdata.models.AutoWindow import PackSeg3DInputs_AutoWindow, ParseLabelDistribution



# --------------------PARAMETERS-------------------- #

# PyTorch
debug    = False                            # 调试模式
use_AMP  = True                             # AMP加速
dist     = False if not debug else False    # 多卡训练总控
use_FSDP = False if not debug else False    # 多卡训练FSDP高级模式
Compile  = True if not debug else False     # torch.dynamo
workers  = 4 if not debug else 0            # DataLoader Worker

# Starting
resume = True
load_from = None
resume_optimizer = True
resume_param_scheduler = True

# Dataset
pre_crop_data_root = '/file1/mgam_datasets/CT_ORG/spacing2_crop64_ccm0.9_npz/'
mha_data_root = '/file1/mgam_datasets/CT_ORG/spacing2_mha'
num_classes = 6
val_sample_ratio = 0.1
wl = 40     # window loacation
ww = 400    # window width
pad_val = 0
seg_pad_val = 0

# Neural Network Hyperparameters
lr = 1e-4
batch_size = 4 if not debug else 2
grad_accumulation = 2 if not debug else 2
embed_dims = 16 if not debug else 8
in_channels = 1
size = (64,64,64)       # 单次前向处理的分辨率, 不限制推理
deep_supervision = True
use_checkpoint = False  # torch.checkpoint

# PMWP Sub-Network Hyperparameters
data_range = [-1024,3072]
num_windows = 8
num_rect = 16
pmwp_lr_mult = 1e-4
TRec_rect_momentum = 0.999
enable_WinE_loss = True
enable_TRec = True
enable_TRec_loss = True
enable_CWF = True

# Training Strategy
iters = 200000 if not debug else 3
logger_interval = 100 if not debug else 1
save_interval = 5000 if not debug else 2
val_on_train = True
val_interval = 100 if not debug else 2
vis_interval = 100
# dynamic_intervals = None
dynamic_intervals = [ # 动态验证间隔
    (5, 100), 
    (150, 1000), 
    (2500, 5000) 
]

# --------------------PARAMETERS-------------------- #
# ////////////////////////////////////////////////// #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# --------------------COMPONENTS-------------------- #

# 数据读取与预处理管线
meta_keys = (
    'img_path', 'seg_map_path', 'ori_shape',
    'img_shape', 'pad_shape', 'scale_factor', 'flip',
    'flip_direction', 'reduce_zero_label', 'series_id')

train_pipeline = [
    dict(type=LoadSampleFromNpz, load_type=['img', 'anno']),
    dict(type=ParseID),
    dict(type=ParseLabelDistribution),
    dict(type=WindowSet, location=wl, width=ww),
    # dict(type=InstanceNorm),
    dict(type=TypeConvert),
    dict(type=PackSeg3DInputs_AutoWindow, meta_keys=meta_keys)
]

val_pipeline = test_pipeline = [
    dict(type=LoadImageFromMHA),
    dict(type=ParseID),
    # dict(type=ParseLabelDistribution),
    dict(type=WindowSet, location=wl, width=ww),
    # dict(type=InstanceNorm),
    dict(type=LoadMaskFromMHA),
    dict(type=TypeConvert),
    dict(type=PackSeg3DInputs_AutoWindow, meta_keys=meta_keys)
]


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers,
    drop_last=False if debug else True,
    pin_memory=True,
    persistent_workers=True if workers > 0 else False,
    sampler=dict(
        type=InfiniteSampler, 
        shuffle=False if debug else True),
    dataset=dict(
        type=CT_ORG_Precrop_Npz,
        split='train',
        data_root_mha=mha_data_root,
        data_root=pre_crop_data_root,
        pipeline=train_pipeline,
        debug=debug,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=workers//2,
    pin_memory=False,
    persistent_workers=True if workers > 0 else False,
    sampler=dict(
        type=RatioSampler, 
        shuffle=False, 
        use_sample_ratio=val_sample_ratio),
    dataset=dict(
        type=CT_ORG_Mha,
        split='val',
        data_root_mha=mha_data_root,
        data_root=mha_data_root,
        pipeline=val_pipeline,
        debug=debug,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=workers//2,
    pin_memory=False,
    persistent_workers=True if workers > 0 else False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=CT_ORG_Mha,
        split='test',
        data_root_mha=mha_data_root,
        data_root=mha_data_root,
        pipeline=test_pipeline,
        debug=debug,
    ),
)

# （不重要）构建评估器
val_evaluator = test_evaluator = dict(
    type=IoUMetric_PerClass, 
    ignore_index=255, 
    iou_metrics=['mIoU','mDice', 'mFscore'], 
    prefix='Perf')

data_preprocessor = dict(
    type=Seg3DDataPreProcessor,
    size=size,
    pad_val=pad_val,
    seg_pad_val=seg_pad_val,
    test_cfg=dict(size=size),
    non_blocking=True,
)

# 训练策略
train_cfg = dict(type=IterBasedTrainLoop,
                 max_iters=iters, 
                 val_interval=val_interval,
                 dynamic_intervals=dynamic_intervals,)
val_cfg  = dict(type=ValLoop, fp16=True)
test_cfg = dict(type=TestLoop)

if not val_on_train:
    val_dataloader = None
    val_evaluator = None
    val_cfg = None

# 优化器
optim_wrapper = dict(
    type=AmpOptimWrapper if use_AMP else OptimWrapper,
    accumulative_counts=grad_accumulation,
    optimizer=dict(type=AdamW,
                   lr=lr,
                   weight_decay=0),
    clip_grad=dict(max_norm=1,
                   norm_type=2,
                   error_if_nonfinite=False),
    # paramwise_cfg=dict(
    #     custom_keys=dict(
    #         pmwp=dict(
    #             decay_mult=0,
    #             lr_mult=10))),
)

# 学习率调整策略
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-3,
        end=iters*0.005,
        by_epoch=False,
    ),
    dict(
        type=PolyLR,
        eta_min=lr*1e-2,
        power=0.6,
        begin=0.3*iters,
        end=0.95*iters,
        by_epoch=False,
    )
]

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(
        type=LoggerJSON,
        interval=logger_interval,
        log_metric_by_epoch=False), 
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook, 
        by_epoch=False, 
        max_keep_ckpts=1,
        interval=save_interval,
        save_best='Perf/mDice' if not debug else None,
        rule='greater' if not debug else None,
        save_last=True if not debug else True),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(
        type=Seg3DVisualizationHook, 
        draw=True, 
        interval=vis_interval if not debug else 1),
)

# torch.dynamo
compile = dict(
    fullgraph=False,
    dynamic=False,
    disable=not Compile,
)

# 分布式训练
runner_type = 'mgam_Runner'
if dist:
    launcher = 'pytorch'
    if use_FSDP:
        strategy = dict(
            type = FSDPStrategy,
            model_wrapper = dict(type=RemasteredFSDP),
        )
    else:
        model_wrapper_cfg=dict(type=RemasteredDDP)
else:
    launcher = 'none'

# 运行环境
env_cfg = dict(
    # 子进程中使用CUDA的话必须使用spawn, 需要保证所有参数可pickle
    # 一般情况下可以使用fork, 可以共享内存空间
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), 
    dist_cfg=dict(backend='nccl'),
    allow_tf32=True,
    benchmark=True,
    allow_fp16_reduced_precision_reduction=True,
    allow_bf16_reduced_precision_reduction=True,
    dynamo_cache_size=8,
    dynamo_supress_errors=False,
    dynamo_logging_level='WARNING',
    torch_logging_level='WARNING',
)

vis_backends = [dict(type=LocalVisBackend), 
                dict(type=TensorboardVisBackend)]
visualizer = dict(
    type=Seg3DLocalVisualizer, 
    vis_backends=vis_backends, 
    name='visualizer',
    alpha=0.5,
    resize=(512,512),
    label_text_scale=0.02,
    label_text_thick=1)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
tta_model = None