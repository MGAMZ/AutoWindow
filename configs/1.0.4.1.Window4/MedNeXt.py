from mmengine.config import read_base
with read_base():
    from .mgam import *

from mgamdata.models.AutoWindow import (
    AutoWindowSetting, ParalleledMultiWindowProcessing, AutoWindowStatusLoggerHook)
from mgamdata.models.MedNeXt import MM_MedNext_Encoder, MM_MedNext_Decoder_3D
from mgamdata.mm.mmseg_Dev3D import DiceLoss_3D

# 神经网络设定
model = dict(
    type = AutoWindowSetting,
    pmwp = dict(
        type=ParalleledMultiWindowProcessing,
        in_channels=in_channels,
        embed_dims=embed_dims,
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
    backbone = dict(
        type=MM_MedNext_Encoder,
        in_channels=in_channels * num_windows, # type: ignore
        embed_dims=embed_dims,
        kernel_size=3,
        dim="3d",
        use_checkpoint=use_checkpoint,
        norm_type='layer',
    ),
    decode_head = dict(
        type=MM_MedNext_Decoder_3D,
        embed_dims=embed_dims,
        kernel_size=3,
        num_classes=num_classes,
        out_channels=num_classes,
        use_checkpoint=use_checkpoint,
        deep_supervision=deep_supervision,
        norm_type='layer',
        ignore_index=0, # 仅对train acc计算有效
        loss_gt_key='gt_sem_seg', # ["gt_sem_seg_one_hot", "gt_sem_seg"]
        loss_decode=dict(
            type=DiceLoss_3D, 
            use_sigmoid=False, 
            ignore_1st_index=False, 
            batch_z=16, 
            # NOTE Severe performance overhead when not being set to None.
            # NOTE Prefer using `ignore_1st_index`.
            # NOTE Invalid Class (Defaults to the last class) has been masked out during preprocess.
            ignore_index=None, 
        ), 
    ), 
    test_cfg=dict(
        mode='slide', 
        crop_size=size, 
        stride=[i//2 for i in size], 
        slide_accumulate_device='cpu'), 
)

custom_hooks = [
    dict(type=AutoWindowStatusLoggerHook, 
         dpi=100, interval=logger_interval),
]