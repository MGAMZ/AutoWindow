from mmengine.config import read_base
with read_base():
    from .mgam import *

from mgamdata.models.AutoWindow import AutoWindowSetting, ParalleledMultiWindowProcessing
from mgamdata.models.MedNeXt import MM_MedNext_Encoder, MM_MedNext_Decoder_3D
from mgamdata.mm.mmseg_Dev3D import DiceLoss_3D

# 神经网络设定
model = dict(
    type = AutoWindowSetting,
    regulation_weight=0.,
    pmwp = dict(
        type=ParalleledMultiWindowProcessing,
        in_channels=in_channels,
        embed_dims=embed_dims,
        num_windows=num_parallel_windows,
        num_rect=8,
        rect_momentum=0.99,
        data_range=[-1024, 3072],
        log_interval=logger_interval,
        enable_VWP=True,
    ),
    backbone = dict(
        type=MM_MedNext_Encoder,
        in_channels=in_channels * num_parallel_windows, # type: ignore
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
        loss_gt_key='gt_sem_seg_one_hot', # ["gt_sem_seg_one_hot", "gt_sem_seg"]
        loss_decode=dict(
            type=DiceLoss_3D,
            use_sigmoid=False,
            ignore_index=None, # NOTE Severe performance overhead when not being set to None.
        ),
    ),
    test_cfg=dict(
        mode='slide',
        crop_size=size,
        stride=size,
        slide_accumulate_device='cpu'),
)