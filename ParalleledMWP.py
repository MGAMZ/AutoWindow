from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from mmengine.model.base_module import BaseModule
from mmseg.registry import MODELS

from mgamdata.mm.mmseg_Dev3D import EncoderDecoder_3D



class value_wise_projector(torch.nn.Module):
    """Value-Wise Projector for one window remapping operation."""
    
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



class PMWP(BaseModule):
    """The top module of Paralleled Multi-Window Processing."""
    
    def __init__(self,
                 in_channels:int,
                 embed_dims:int,
                 window_embed_dims:int=32,
                 num_windows:int=4,
                 num_bins:int=512,
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
        self.num_windows = num_windows
        self.num_bins = num_bins
        self.remap_range = remap_range
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self._init_PMWP()


    def _init_PMWP(self):
        for i in range(self.num_windows):
            setattr(self, f"pmwp_window_{i}", 
                    value_wise_projector(
                        in_channels=self.in_channels,
                        nbins=self.num_bins,
                        remap_range=self.remap_range,
                        dim=self.dim))
        
        # TODO Maybe Point-Wise Attention?
        self.window_fusion_weight = torch.nn.Parameter(
            torch.ones(self.num_windows, self.num_windows))


    def forward_PMWP(self, inputs:Sequence[Tensor]):
        """
        Args:
            inputs (Sequence[Tensor]): 
                Each element is a tensor of shape (N, C, ...).
                The length of the list is num_windows.
        """
        pmwp_embedded = []
        
        for i, window in enumerate(inputs):
            projected = getattr(self, f"pmwp_window_{i}")(window)
            pmwp_embedded.append(projected)
        pmwp_embedded = torch.stack(pmwp_embedded, dim=0) # [W, N, C, ...]
        ori_shape = pmwp_embedded.shape
        
        # Cross-Window Fusion
        fused = torch.matmul(
            self.window_fusion_weight, 
            pmwp_embedded.reshape(ori_shape[0], -1)
        ).reshape(ori_shape)
        
        # Window Concatenate
        window_concat_on_channel = fused.transpose(0,1).reshape(
            ori_shape[0], ori_shape[1]*ori_shape[2], *ori_shape[3:])
        
        return window_concat_on_channel # [N, Win*C, ...]


class PMWP_Warpper(EncoderDecoder_3D):
    def extract_feat(self, inputs:Tensor):
        # inputs: [N, C, ...]
        pmwp_out = self.checkpoint(self.forward_PMWP, inputs)
        x = self.checkpoint(self.backbone, pmwp_out)
        if self.with_neck:
            x = self.checkpoint(self.neck, x)
        return x
