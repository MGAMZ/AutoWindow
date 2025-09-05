from mmengine.config import read_base
with read_base():
    from .mgam import *

from src.model import ParalleledMultiWindowProcessing, AutoWindowStatusLoggerHook, AutoWindowLite
from itkit.models.UNETR import UNETR
from itkit.mm.mmseg_Dev3D import DiceLoss_3D

# custom_hooks = [
#     dict(type=AutoWindowStatusLoggerHook,
#          dpi=100,
#          interval=logger_interval),
# ]

# 神经网络设定
model = dict(
    type=AutoWindowLite,
    pmwp=dict(
        type=ParalleledMultiWindowProcessing,
        in_channels=in_channels,
        num_windows=num_windows,
        num_rect=num_rect,
        TRec_rect_momentum=TRec_rect_momentum,
        data_range=data_range,
        enable_WinE_loss=enable_WinE_loss,
        enable_TRec=enable_TRec,
        enable_TRec_loss=enable_TRec_loss,
        enable_CWF=enable_CWF,
        lr_mult=pmwp_lr_mult,
    ),
    binary_segment_threshold=None,
    auto_activate_after_logits=True,
    inference_PatchSize=size,
    inference_PatchStride=[s//2 for s in size],
    inference_PatchAccumulateDevice='cpu',
    backbone=dict(
        type=UNETR,
        in_channels=in_channels*num_windows, # pyright: ignore
        out_channels=num_classes,
        img_size=size,
        use_flash_attention=True,
    ),
    criterion=dict(
        type=DiceLoss_3D,
        batch_z=None,
        ignore_1st_index=False,
        ignore_index=None,
    )
)