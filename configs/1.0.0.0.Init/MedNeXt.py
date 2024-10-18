from mmengine.config import read_base
with read_base():
    from .mgam import *

from mmseg.models.segmentors import EncoderDecoder
from mmseg.models.losses import DiceLoss

from mgamdata.models.MedNeXt import MM_MedNext_Encoder, MM_MedNext_Decoder_Vallina


# 神经网络设定
model = dict(
    type = EncoderDecoder,
    backbone = dict(
        type=MM_MedNext_Encoder,
        in_channels=in_channels,
        embed_dims=embed_dims,
    ),
    decode_head = dict(
        type=MM_MedNext_Decoder_Vallina,
        embed_dims=embed_dims,
        num_classes=num_classes,
        out_channels=num_classes,
        threshold=0.3,
        norm_cfg=None,
        align_corners=False,
        ignore_index=255,
        loss_decode=dict(type=DiceLoss, 
                         use_sigmoid=False,
                         ignore_index=255),
    ),
    test_cfg=dict(mode='slide', 
                  crop_size=size,
                  stride=[i//3 for i in size]),
)