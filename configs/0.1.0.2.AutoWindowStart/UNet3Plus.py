from mmengine.config import read_base
with read_base():
    from .mgam import *

from mgamdata.models.AutoWindow import ParalleledMultiWindowProcessing, AutoWindowStatusLoggerHook, AutoWindowLite
from mgamdata.models.UNet3Plus import UNet3Plus
from mgamdata.mm.mmseg_Dev3D import DiceLoss_3D

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
        type=UNet3Plus,
        input_shape=(in_channels*num_windows, *size), # pyright: ignore
        output_channels=num_classes,
        filters=[16, 32, 32, 32, 64],
        dim=3,
        use_torch_checkpoint=True
    ),
    criterion=dict(
        type=DiceLoss_3D,
        batch_z=None,
        ignore_1st_index=False,
        ignore_index=None,
    )
)