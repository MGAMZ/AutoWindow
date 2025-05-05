from mmengine.config import read_base
with read_base():
    from .mgam import *

from mgamdata.models.AutoWindow import ParalleledMultiWindowProcessing, AutoWindowStatusLoggerHook, AutoWindowLite
from mgamdata.models.SegFormer3D.Remastered import SegFormer3D
from mgamdata.mm.mmseg_Dev3D import DiceLoss_3D

custom_hooks = [
    dict(type=AutoWindowStatusLoggerHook,
         dpi=100,
         interval=logger_interval),
] if num_windows is not None else []

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
    ) if num_windows is not None else None,
    num_classes=num_classes,
    binary_segment_threshold=None,
    inference_PatchSize=size,
    inference_PatchStride=[s//2 for s in size],
    inference_PatchAccumulateDevice='cuda',
    backbone=dict(
        type=SegFormer3D,
        in_channels=in_channels if num_windows is None else in_channels*num_windows, # pyright: ignore
        num_classes=num_classes,
        embed_dims=[128, 256, 512, 1024],
        num_heads=[4, 8, 16, 32],
        depths=[2, 2, 2, 2],
        mlp_ratios=[2, 2, 2, 2],
        sr_ratios=[2, 2, 2, 1],
        patch_kernel_size=[7, 3, 3, 3],
        patch_stride=[3, 2, 2, 2],
        patch_padding=[2, 1, 1, 1],
        decoder_head_embedding_dim=128,
    ),
    criterion=dict(
        type=DiceLoss_3D,
        split_Z=False,
        to_onehot_y=True,
        sigmoid=False,
        softmax=True,
        squared_pred=True,
        include_background=True,
    )
)