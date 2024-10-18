from mmengine.config import read_base
with read_base():
    from ..base import *

from torch.optim.adamw import AdamW

from mmengine.runner import IterBasedTrainLoop
from mmengine.optim.scheduler import CosineAnnealingLR
from mmengine._strategy import FSDPStrategy
from mmengine.optim import OptimWrapper, AmpOptimWrapper
from mmengine.dataset.sampler import DefaultSampler, InfiniteSampler
from mmseg.datasets.transforms import PackSegInputs, RandomRotate, RandomCrop, RandomFlip, Resize
from mmseg.models.data_preprocessor import SegDataPreProcessor

from mgamdata.dataset.Totalsegmentator.mm_dataset import TotalsegmentatorSegDataset
from mgamdata.mm.mmseg_PlugIn import IoUMetric_PerClass
from mgamdata.mm.mmeng_PlugIn import RemasteredDDP, RemasteredFSDP
from mgamdata.process.GeneralPreProcess import WindowSet, TypeConvert, RandomRoll
from mgamdata.process.LoadBiomedicalData import LoadImgFromOpenCV, LoadAnnoFromOpenCV


# 环境
debug    = False                             # 调试模式
use_AMP  = True                             # AMP加速
dist     = False if not debug else False     # 多卡训练总控
use_FSDP = False if not debug else False    # 多卡训练FSDP高级模式
Compile  = True if not debug else False     # torch.dynamo
workers  = 0 if debug else 4                # DataLoader Worker

# 窗宽位
wl = 40     # 窗位 40-60     Optimum: 40
ww = 400    # 窗宽 300-400   Optimum: 400

# 神经网络超参
lr = 1e-4
batch_size = 4
embed_dims = 16
in_channels = 1
num_classes = 5
size = (512,512)    # 单次前向处理的分辨率, 不限制推理
use_checkpoint = False  # torch.checkpoint

# 流程控制
iters = 500000 if not debug else 3
logger_interval = 200 if not debug else 1
save_interval = 5000 if not debug else 2
val_on_train = True
val_interval = 1 if not debug else 2
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
train_pipeline = [
    dict(type=LoadImgFromOpenCV),
    dict(type=LoadAnnoFromOpenCV),
    dict(type=Resize, scale=size),
    dict(type=RandomRoll, direction=['horizontal', 'vertical'], gap=[256, 256],
         erase=False, pad_val=-1000, seg_pad_val=0),
    dict(type=RandomRotate, prob=1.0, degree=180, pad_val=-1000, seg_pad_val=0),
    dict(type=RandomFlip, prob=[0.5, 0.5], direction=['horizontal', 'vertical']),
    dict(type=WindowSet, location=wl, width=ww),
    dict(type=TypeConvert),
    dict(type=PackSegInputs)
]
val_pipeline = test_pipeline = [
    dict(type=LoadImgFromOpenCV),
    dict(type=Resize, scale=size),
    dict(type=WindowSet, location=wl, width=ww),
    dict(type=LoadAnnoFromOpenCV),
    dict(type=TypeConvert),
    dict(type=PackSegInputs)
]

# （不重要）构建dataloader
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers,
    drop_last=False if debug else True,
    pin_memory=False,
    persistent_workers=True if workers > 0 else False,
    sampler=dict(type=InfiniteSampler, shuffle=False if debug else True),
    dataset=dict(
        type=TotalsegmentatorSegDataset,
        split='train',
        pipeline = train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers//2,
    pin_memory=False,
    persistent_workers=True if workers > 0 else False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=TotalsegmentatorSegDataset,
        split='val',
        pipeline=test_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers//2,
    pin_memory=False,
    persistent_workers=True if workers > 0 else False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=TotalsegmentatorSegDataset,
        split='test',
        pipeline=test_pipeline,
    ),
)

# （不重要）构建评估器
val_evaluator = test_evaluator = dict(
    type=IoUMetric_PerClass, 
    ignore_index=255, 
    iou_metrics=['mIoU','mDice'], 
    prefix='Perf')
if not val_on_train:
    val_dataloader = None
    val_evaluator = None
    val_cfg = None

# （不重要）MM框架数据传输与规范化中间件
data_preprocessor = dict(
    type=SegDataPreProcessor,
    size=size,
)

# 训练策略
train_cfg = dict(type=IterBasedTrainLoop,
                 max_iters=iters, 
                 val_interval=val_interval,
                 dynamic_intervals=dynamic_intervals,)

# 优化器
optim_wrapper = dict(
    type=AmpOptimWrapper if use_AMP else OptimWrapper, 
    optimizer=dict(type=AdamW,
                   lr=lr,
                   weight_decay=1e-2),
    clip_grad=dict(max_norm=1, 
                   norm_type=2, 
                   error_if_nonfinite=False)
)

# 学习率调整策略
param_scheduler = [
    dict(
        type=CosineAnnealingLR,
        T_max=iters,
        eta_min_ratio=0.05,
        by_epoch=False,
        begin=0,
        end=iters*0.9)
]

# （不重要）Hooks
default_hooks.update(   # type: ignore
    checkpoint=dict(
        interval=save_interval,
        save_best='Perf/mDice' if not debug else None,
        rule='greater' if not debug else None,
        save_last=True if not debug else True),
    logger=dict(interval=logger_interval),
    visualization=dict(interval=1000 if not debug else 1))

# torch.dynamo
compile = dict(
    fullgraph=False,
    dynamic=False,
    disable=not Compile,
)

# 分布式训练
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

# 断点续训
resume=True
