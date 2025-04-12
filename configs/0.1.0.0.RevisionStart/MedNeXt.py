from mmengine.config import read_base
with read_base():
    from .mgam import *

from mgamdata.models.AutoWindow import AutoWindowSetting, ParalleledMultiWindowProcessing, AutoWindowStatusLoggerHook
from mgamdata.models.MedNeXt import MedNeXt
from mgamdata.mm.mmseg_Dev3D import DiceLoss_3D
from mgamdata.mm.mgam_models import mgam_Seg3D_Lite

# custom_hooks = [
#     dict(type=AutoWindowStatusLoggerHook, 
#          dpi=100, interval=logger_interval),
# ]

embed_dims = 12
MedNeXt_Checkpoint = True

# 神经网络设定
model = dict(
    type = mgam_Seg3D_Lite,
    # pmwp = dict(
    #     type=ParalleledMultiWindowProcessing,
    #     in_channels=in_channels,
    #     embed_dims=embed_dims,
    #     num_windows=num_windows,
    #     num_rect=num_rect,
    #     TRec_rect_momentum=TRec_rect_momentum,
    #     data_range=data_range,
    #     enable_WinE_loss=enable_WinE_loss,
    #     enable_TRec=enable_TRec,
    #     enable_TRec_loss=enable_TRec_loss,
    #     enable_CWF=enable_CWF,
    #     lr_mult=pmwp_lr_mult,
    # ),
    binary_segment_threshold=None,
    inference_PatchSize=size,
    inference_PatchStride=[s//2 for s in size],
    inference_PatchAccumulateDevice='cpu',
    backbone = dict(
        type=MedNeXt,
        in_channels=in_channels, # type: ignore
        n_channels=embed_dims,
        kernel_size=5,
        n_classes=num_classes,
        checkpoint_style='outside_block' if MedNeXt_Checkpoint else None,
        dim="3d"
    ),
    criterion=dict(
        type=DiceLoss_3D,
        batch_z=None,
        ignore_1st_index=False,
        ignore_index=None,
    )
)