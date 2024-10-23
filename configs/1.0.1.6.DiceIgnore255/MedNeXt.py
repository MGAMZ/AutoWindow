from mmengine.config import read_base
with read_base():
    from .mgam import *

from mgamdata.models.MedNeXt import MM_MedNext_Encoder, MM_MedNext_Decoder_3D
from mgamdata.mm.mmseg_Dev3D import EncoderDecoder_3D, DiceLoss_3D

# 神经网络设定
model = dict(
    type = EncoderDecoder_3D,
    backbone = dict(
        type=MM_MedNext_Encoder,
        in_channels=in_channels,
        embed_dims=embed_dims,
        kernel_size=3,
        dim="3d",
        use_checkpoint=use_checkpoint,
    ),
    decode_head = dict(
        type=MM_MedNext_Decoder_3D,
        embed_dims=embed_dims,
        kernel_size=3,
        num_classes=num_classes,
        out_channels=num_classes,
        use_checkpoint=use_checkpoint,
        deep_supervision=deep_supervision,
        ignore_index=0, # 仅对train acc计算有效
        loss_decode=dict(
            type=DiceLoss_3D,
            use_sigmoid=False,
            ignore_index=255),
    ),
    test_cfg=dict(
        mode='slide',
        crop_size=size,
        stride=size,
        slide_accumulate_device='cpu'),
)