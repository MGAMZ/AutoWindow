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
from mmengine.optim.scheduler import CosineAnnealingLR
from mmengine._strategy import FSDPStrategy
from mmengine.optim import OptimWrapper, AmpOptimWrapper
from mmengine.dataset.sampler import DefaultSampler, InfiniteSampler
from mmengine.visualization import LocalVisBackend
from mmengine.visualization import TensorboardVisBackend

# customize
from mgamdata.mm.mmeng_PlugIn import LoggerJSON
from mgamdata.mm.mmseg_PlugIn import IoUMetric_PerClass
from mgamdata.mm.mmeng_PlugIn import RemasteredDDP, RemasteredFSDP, RatioSampler
from mgamdata.process.GeneralPreProcess import WindowSet, TypeConvert, InstanceNorm
from mgamdata.process.LoadBiomedicalData import LoadImageFromMHA, LoadMaskFromMHA
from mgamdata.dataset.Totalsegmentator.mm_dataset import TotalsegmentatorSeg3DDataset, ParseID
from mgamdata.mm.mmseg_Dev3D import (
    PackSeg3DInputs, Resize3D, RandomCrop3D,
    Seg3DDataPreProcessor, Seg3DLocalVisualizer, Seg3DVisualizationHook,
)




# --------------------PARAMETERS-------------------- #
debug    = False                            # 调试模式
use_AMP  = True                             # AMP加速
dist     = True if not debug else False     # 多卡训练总控
use_FSDP = False if not debug else False    # 多卡训练FSDP高级模式
Compile  = True if not debug else False     # torch.dynamo
workers  = 0 if debug else 4                # DataLoader Worker

# Totalsegmentator Dataset
data_root = '/home/zhangyq.sx/Totalsegmentator_Data/Totalsegmentator_dataset_v201/spacing_1_mha'
subset = 'organ' # ['organ', None]
num_classes = 119 if subset is None else len(CLASS_SUBSET_MAP[subset])
val_sample_ratio = 0.1
wl = 193    # window loacation
ww = 800    # window width
pad_val = -2000
seg_pad_val = 0

# 神经网络超参
lr = 5e-4
batch_size = 1 if not debug else 2
grad_accumulation = 2
embed_dims = 32
in_channels = 1
size = (128,128,128)       # 单次前向处理的分辨率, 不限制推理
use_checkpoint = False  # torch.checkpoint

# 流程控制
iters = 500000 if not debug else 3
logger_interval = 200 if not debug else 1
save_interval = 5000 if not debug else 2
val_on_train = True
val_interval = 2000 if not debug else 2
dynamic_intervals = None
# dynamic_intervals = [   # 动态验证间隔
#     (5, 5),
#     (50, 10),
#     (100, 25),
#     (300, 100),
#     (1000, 250),
#     (3000, 500),
#     (5000, 1000),
#     (20000, 5000)
# ]



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
    dict(type=LoadImageFromMHA),
    dict(type=ParseID),
    dict(type=LoadMaskFromMHA),
    dict(type=RandomCrop3D, crop_size=size),
    dict(type=WindowSet, location=wl, width=ww),
    dict(type=InstanceNorm),
    dict(type=TypeConvert),
    dict(type=PackSeg3DInputs, meta_keys=meta_keys)
]

val_pipeline = test_pipeline = [
    dict(type=LoadImageFromMHA),
    dict(type=ParseID),
    dict(type=WindowSet, location=wl, width=ww),
    dict(type=InstanceNorm),
    dict(type=LoadMaskFromMHA),
    dict(type=TypeConvert),
    dict(type=PackSeg3DInputs, meta_keys=meta_keys)
]

# （不重要）构建dataloader
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers,
    drop_last=False if debug else True,
    pin_memory=False,
    persistent_workers=True if workers > 0 else False,
    sampler=dict(
        type=InfiniteSampler, 
        shuffle=False if debug else True),
    dataset=dict(
        type=TotalsegmentatorSeg3DDataset,
        split='train',
        data_root=data_root,
        subset=subset,
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
        type=TotalsegmentatorSeg3DDataset,
        split='val',
        data_root=data_root,
        subset=subset,
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
        type=TotalsegmentatorSeg3DDataset,
        split='test',
        data_root=data_root,
        subset=subset,
        pipeline=test_pipeline,
        debug=debug,
    ),
)

# （不重要）构建评估器
val_evaluator = test_evaluator = dict(
    type=IoUMetric_PerClass, 
    ignore_index=255, 
    iou_metrics=['mIoU','mDice'], 
    prefix='Perf')

data_preprocessor = dict(
    type=Seg3DDataPreProcessor,
    size=size,
    pad_val=pad_val,
    seg_pad_val=seg_pad_val,
    test_cfg=dict(size=size),
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
                   weight_decay=1e-2),
    clip_grad=dict(max_norm=1, 
                   norm_type=2, 
                   error_if_nonfinite=False)
)

# 学习率调整策略
param_scheduler = []


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
        interval=100 if not debug else 1),
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
    dynamo_cache_size=2,
    dynamo_supress_errors=False,
    dynamo_logging_level='ERROR',
    torch_logging_level='ERROR',
)

vis_backends = [dict(type=LocalVisBackend), 
                dict(type=TensorboardVisBackend)]
visualizer = dict(
    type=Seg3DLocalVisualizer, 
    vis_backends=vis_backends, 
    name='visualizer',
    alpha=0.2,
    resize=(512,512))
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = None
resume=True