from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from mmengine.model.base_module import BaseModule
from mmseg.registry import MODELS

from mgamdata.mm.mmseg_Dev3D import EncoderDecoder_3D



class WindowExtractor(BaseModule):
    "Extract values from raw array using a learnable window."

    def __init__(self, 
                 value_range:list[float],
                 window_width:int,):
        self.window_location = torch.nn.Parameter(
            torch.tensor([value_range[0], value_range[1]]))
    
    @property
    def current_window(self):
        return self.window_location.detach().cpu().numpy()
    
    
    def forward(self, inputs:Tensor):
        """
        Args:
            inputs (Tensor): (N, ...)
        """
        lower_bound = self.window_location[0]
        upper_bound = self.window_location[1]
        clipped = torch.clamp(inputs, lower_bound, upper_bound)
        return clipped



class ValueWiseProjector(BaseModule):
    """
    Value-Wise Projector for one window remapping operation.
    The extracted value are fine-tuned by this projector.
    """
    
    def __init__(self, 
                 in_channels:int, 
                 nbins:int, 
                 remap_range:list[float]=[0,1], 
                 dim:str='3d'):
        super().__init__()
        self.in_channels = in_channels
        self.nbins = nbins
        self.remap_range = remap_range
        self.dim = dim
        
        if dim.lower() == '2d':
            self.pmwm_norm = torch.nn.InstanceNorm2d(
                num_features=in_channels,
                affine=True, 
                track_running_stats=True)
        else:
            self.pmwm_norm = torch.nn.InstanceNorm3d(
                num_features=in_channels,
                affine=True,
                track_running_stats=True)
        
        self.projection_map = torch.nn.Parameter(
            torch.linspace(
                start=remap_range[0],
                end=remap_range[1],
                step=nbins))
    
    @property
    def current_projection(self):
        return self.projection_map.detach().cpu().numpy()
    
    
    def regulation(self):
        """
        Limit the projector ability to ensure it's behavior,
        which aligns with the physical meaning.
        """
        
        # Ascending projected value along the index.
        index = torch.arange(len(self.projection_map)).float()
        target_map = index / \
                     (len(self.projection_map) - 1) * \
                     (self.projection_map[-1] - self.projection_map[0]) + \
                     self.projection_map[0]
        ascend_loss = torch.sum((self.projection_map - target_map) ** 2)
        
        # More smoothness around the center.
        diff = torch.diff(self.projection_map)
        center = len(self.projection_map) // 2
        smoothness_loss = torch.sum(diff[:center] ** 2) + \
                          torch.sum(diff[center:] ** 2)
        
        return ascend_loss + smoothness_loss
    
    
    def forward(self, inputs:Tensor):
        """
        Args:
            inputs (Tensor): (N, C, ...)
        """
        normed_x = self.pmwm_norm(inputs)
        scaled_x = normed_x * (self.nbins - 1)
        lower_indices = scaled_x.floor().long()
        upper_indices = scaled_x.ceil().long()
        lower_values = self.projection_map[lower_indices]
        upper_values = self.projection_map[upper_indices]
        
        # interpolate between bins
        weights = (scaled_x - lower_indices)
        projected = lower_values * (1 - weights) + upper_values * weights
        
        return projected



class BatchCrossWindowFusion(BaseModule):
    def __init__(self, num_windows:int):
        super().__init__()
        self.window_fusion_weight = torch.nn.Parameter(
            torch.ones(num_windows, num_windows))
    
    @property
    def current_fusion(self):
        return self.window_fusion_weight.detach().cpu().numpy()
    
    
    def forward(self, inputs:Tensor):
        """
        Args:
            inputs (Tensor): (Win, N, C, ...)
        
        Returns:
            Tensor: (N, Win*C, ...)
        """
        ori_shape = inputs.shape
        fused = torch.matmul(
            self.window_fusion_weight, 
            inputs.reshape(ori_shape[0], -1)
        ).reshape(ori_shape)
        
        # Window Concatenate
        window_concat_on_channel = fused.transpose(0,1).reshape(
            ori_shape[1], ori_shape[0]*ori_shape[2], *ori_shape[3:])
        
        return window_concat_on_channel # [N, Win*C, ...]



class ParalleledMultiWindowProcessing(BaseModule):
    """The top module of Paralleled Multi-Window Processing."""
    
    def __init__(self,
                 in_channels:int,
                 embed_dims:int,
                 window_embed_dims:int=32,
                 window_width:int=200,
                 num_windows:int=4,
                 num_bins:int=512,
                 data_range:list[float]=[-1024, 3072],
                 remap_range:list[float]=[0,1],
                 dim='3d',
                 use_checkpoint:bool=False
                ):
        assert dim.lower() in ['2d', '3d']
        super().__init__()
        if use_checkpoint:
            self.checkpoint = lambda f,x: checkpoint(f, x, use_reentrant=False)
        else:
            self.checkpoint = lambda f,x: f(x)
        
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.window_embed_dims = window_embed_dims
        self.window_width = window_width
        self.num_windows = num_windows
        self.num_bins = num_bins
        self.data_range = data_range
        self.remap_range = remap_range
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self._init_PMWP()


    def _init_PMWP(self):
        for i in range(self.num_windows):
            setattr(self, f"window_extractor_{i}", 
                WindowExtractor(
                    value_range=self.data_range,
                    window_width=self.window_width))
            setattr(self, f"value_wise_projector_{i}", 
                ValueWiseProjector(
                    in_channels=self.in_channels,
                    nbins=self.num_bins,
                    remap_range=self.remap_range,
                    dim=self.dim))
        
        # TODO Maybe Point-Wise Attention?
        self.cross_window_fusion = BatchCrossWindowFusion(self.num_windows)


    def forward_PMWP(self, inputs:Sequence[Tensor]):
        """
        Args:
            inputs (Sequence[Tensor]): 
                Each element is a tensor of shape (N, C, ...).
                The length of the list is num_windows.
        """
        x = []
        
        for i, window in enumerate(inputs):
            extracted = getattr(self, f"window_extractor_{i}")(window)
            projected = getattr(self, f"value_wise_projector_{i}")(extracted)
            x.append(projected)
        x = torch.stack(x, dim=0) # [W, N, C, ...]
        
        x = self.cross_window_fusion(x) # [N, Win*C, ...]
        
        return x # [N, Win*C, ...]



class AutoWindowSettingWarpper(EncoderDecoder_3D):
    "Compatible Plugin for Auto Window Setting."
    
    def __init__(self, pmwp:dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pmwp = MODELS.build(pmwp)
    
    
    def extract_feat(self, inputs:Tensor):
        # inputs: [N, C, ...]
        # pmwp_out: [N, num_window * C, ...]
        # TODO Downsampling Channel?
        pmwp_out = self.checkpoint(self.pmwp, inputs)
        return super().extract_feat(pmwp_out)
