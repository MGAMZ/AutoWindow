from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW
from torch.distributed.fsdp.api import ShardingStrategy

# mmengine
from mmengine.runner import ValLoop
from mmengine.runner import TestLoop
from mmengine.hooks.iter_timer_hook import IterTimerHook
from mmengine.hooks.param_scheduler_hook import ParamSchedulerHook
from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.hooks import DistSamplerSeedHook
from mmengine.runner import IterBasedTrainLoop
from mmengine.model.wrappers import MMFullyShardedDataParallel
from mmengine.optim.scheduler import LinearLR, PolyLR
from mmengine.optim import OptimWrapper, AmpOptimWrapper
from mmengine._strategy.deepspeed import DeepSpeedOptimWrapper, DeepSpeedStrategy
from mmengine.dataset.sampler import DefaultSampler, InfiniteSampler
from mmengine.visualization import TensorboardVisBackend

# customize
from mgamdata.mm.mmseg_PlugIn import IoUMetric_PerClass
from mgamdata.mm.mmeng_PlugIn import RemasteredDDP, RatioSampler, LoggerJSON, mgam_OptimWrapperConstructor, RemasteredFSDP_Strategy
from mgamdata.mm.mmseg_Dev3D import Seg3DDataPreProcessor
from mgamdata.mm.visualization import SegViser, BaseVisHook, LocalVisBackend
from mgamdata.process.GeneralPreProcess import WindowSet, InstanceNorm
from mgamdata.process.LoadBiomedicalData import LoadImageFromMHA, LoadMaskFromMHA, LoadCTPreCroppedSampleFromNpz
from mgamdata.dataset.Totalsegmentator import Tsd3D_PreCrop_Npz, Tsd_Mha, TSD_CLASS_INDEX_MAP_GENERAL_REDUCTED
from mgamdata.models.AutoWindow import PackSeg3DInputs_AutoWindow, ParseLabelDistribution



# --------------------PARAMETERS-------------------- #

# PyTorch
debug = False   # 调试模式
use_AMP = True  # AMP加速
dist = True if not debug else False  # 分布式使能
MP_mode = "ddp"  # 分布式计算模式 Literal[`"ddp", "fsdp", "deepspeed"]
Compile = True if not debug else False  # torch.dynamo
workers = 4 if not debug else 0  # DataLoader Worker

# Starting
resume = True
load_from = None
resume_optimizer = True
resume_param_scheduler = True

# Dataset
pre_crop_data_root = '/zyq_local/mgam_datasets/Totalsegmentator/spacingZ2_sizeXY256_cropZ16_npz/'
mha_data_root = '/zyq_local/mgam_datasets/Totalsegmentator/spacingZ2_sizeXY256_mha/'
tsd_meta = '/zyq_remote/mgam_datasets/Totalsegmentator/meta_v2.csv'
num_classes = 45
val_sample_ratio = 1.0 if not debug else 0.1
wl = 50     # window loacation
ww = 400    # window width
pad_val = 0
seg_pad_val = 0

# Neural Network Hyperparameters
lr = 1e-4
batch_size = 4
grad_accumulation = 1
weight_decay = 0
in_channels = 1
size = (16,256,256)

# PMWP Sub-Network Hyperparameters
data_range = [-1024,3072]
num_windows = None
num_rect = 8
pmwp_lr_mult = None
TRec_rect_momentum = 0.999
enable_WinE_loss = False
enable_TRec = True
enable_TRec_loss = False
enable_CWF = True

# Training Strategy
iters = 500000 if not debug else 3
logger_interval = 100 if not debug else 1
save_interval = 5000 if not debug else 2
val_on_train = True
val_interval = 500 if not debug else 2
vis_interval = 100
# dynamic_intervals = None
dynamic_intervals = [ # 动态验证间隔
    (250, 500),
    (2000, 1000),
    (5000, 5000)
]

# --------------------PARAMETERS-------------------- #
# ////////////////////////////////////////////////// #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# --------------------COMPONENTS-------------------- #

# 数据读取与预处理管线
train_pipeline = [
    dict(type=LoadCTPreCroppedSampleFromNpz, load_type=['img', 'anno']),
    dict(type=ParseLabelDistribution),
    dict(type=WindowSet, level=wl, width=ww),
    # dict(type=InstanceNorm),
    dict(type=PackSeg3DInputs_AutoWindow)
]

val_pipeline = test_pipeline = [
    dict(type=LoadImageFromMHA),
    dict(type=LoadMaskFromMHA),
    dict(type=ParseLabelDistribution),
    dict(type=WindowSet, level=wl, width=ww),
    # dict(type=InstanceNorm),
    dict(type=PackSeg3DInputs_AutoWindow)
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
        type=Tsd3D_PreCrop_Npz,
        split='train',
        meta_csv=tsd_meta,
        data_root=pre_crop_data_root,
        class_reduction=TSD_CLASS_INDEX_MAP_GENERAL_REDUCTED,
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
        type=Tsd_Mha,
        split='val',
        data_root=mha_data_root,
        class_reduction=TSD_CLASS_INDEX_MAP_GENERAL_REDUCTED,
        meta_csv=tsd_meta,
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
        type=Tsd_Mha,
        split='test',
        data_root=mha_data_root,
        class_reduction=TSD_CLASS_INDEX_MAP_GENERAL_REDUCTED,
        meta_csv=tsd_meta,
        pipeline=test_pipeline,
        debug=debug,
    ),
)

# 构建评估器
val_evaluator = test_evaluator = dict(
    type=IoUMetric_PerClass,
    ignore_index=255,
    iou_metrics=['mIoU','mDice', 'mFscore'],
    prefix='Perf')

data_preprocessor = dict(
    type=Seg3DDataPreProcessor,
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
if MP_mode == "deepspeed" and dist:
    optim_wrapper = dict(
        type=DeepSpeedOptimWrapper,
        optimizer=dict(type=AdamW, lr=lr, weight_decay=weight_decay),
        accumulative_counts=grad_accumulation,
        constructor=dict(type=mgam_OptimWrapperConstructor),
    )
else:
    optim_wrapper = dict(
        type=AmpOptimWrapper if use_AMP else OptimWrapper,
        accumulative_counts=grad_accumulation,
        optimizer=dict(type=AdamW, lr=lr, weight_decay=weight_decay),
        clip_grad=dict(max_norm=5, norm_type=2, error_if_nonfinite=False),
        constructor=dict(type=mgam_OptimWrapperConstructor),
    )
if use_AMP and dist and MP_mode=='fsdp':
    optim_wrapper["use_fsdp"] = True

# 学习率调整策略
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-3,
        end=iters*0.1,
        by_epoch=False,
    ),
    dict(
        type=PolyLR,
        eta_min=lr*1e-2,
        power=0.6,
        begin=0.5*iters,
        end=0.95*iters,
        by_epoch=False,
    )
] if not debug else []

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
        type=BaseVisHook,
        val_vis_interval=vis_interval if not debug else 1,
        test_vis_interval=vis_interval if not debug else 1),
)

visualizer = dict(
    type=SegViser,
    vis_backends=[dict(type=LocalVisBackend), dict(type=TensorboardVisBackend)],
    dim=3)

# torch.dynamo
compile = dict(
    fullgraph=False,
    dynamic=False,
    disable=not Compile,
)

# 分布式训练
runner_type = "mgam_Runner"
if dist:
    launcher = "pytorch"
    if MP_mode == "deepspeed":
        strategy = dict(
            type=DeepSpeedStrategy,
            fp16=dict(
                enabled=True,
                auto_cast=True,
                fp16_master_weights_and_grads=False,
                loss_scale=0,
                loss_scale_window=500,
                hysteresis=2,
                min_loss_scale=1,
                initial_scale_power=15,
            ),
            inputs_to_half=None,
            zero_optimization=dict(
                stage=3,
                allgather_partitions=True,
                reduce_scatter=True,
                allgather_bucket_size=5e7,
                reduce_bucket_size=5e7, # 1e6 available
                overlap_comm=True,
                contiguous_gradients=True,
                cpu_offload=False,
                ignore_unused_parameters=True,
                stage3_gather_16bit_weights_on_model_save=True),
        )
    elif MP_mode == "ddp":
        model_wrapper_cfg = dict(type=RemasteredDDP)
    elif MP_mode == "fsdp":
        strategy = dict(
            type=RemasteredFSDP_Strategy,
            model_wrapper=dict(
                type=MMFullyShardedDataParallel, 
                use_orig_params=True, 
                sharding_strategy=ShardingStrategy.FULL_SHARD,
            ),
        )
else:
    launcher = "none"

# 运行环境
env_cfg = dict(
    # 子进程中使用CUDA的话必须使用spawn, 需要保证所有参数可pickle
    # 一般情况下可以使用fork, 可以共享内存空间
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
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
log_processor = dict(by_epoch=False)
log_level = 'INFO'