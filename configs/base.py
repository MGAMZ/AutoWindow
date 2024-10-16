# mmsegmentation
from mmseg.visualization import SegLocalVisualizer
from mmseg.engine.hooks import SegVisualizationHook

# mmengine
from mmengine.runner import ValLoop
from mmengine.runner import TestLoop
from mmengine.hooks.iter_timer_hook import IterTimerHook
from mmengine.hooks.param_scheduler_hook import ParamSchedulerHook
from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.hooks import DistSamplerSeedHook

from mmengine.visualization import LocalVisBackend
from mmengine.visualization import TensorboardVisBackend

# customize
from mgamdata.mm.mmeng_PlugIn import LoggerJSON




# Task Control
val_cfg  = dict(type=ValLoop, fp16=True)
test_cfg = dict(type=TestLoop)
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerJSON, 
                interval=100, log_metric_by_epoch=False), 
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, 
                    by_epoch=False, 
                    max_keep_ckpts=1),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook, 
                       draw=True),
)

# runtime
runner_type = 'mgam_Runner'
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
    type=SegLocalVisualizer, 
    vis_backends=vis_backends, 
    name='visualizer',
    alpha=0.2)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = None
