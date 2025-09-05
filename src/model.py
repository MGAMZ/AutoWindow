import pdb
from collections import OrderedDict
from collections.abc import Callable, Sequence
from io import BytesIO
from PIL import Image
from functools import partial

import seaborn
import numpy as np
import torch
from torch import Tensor
from matplotlib import pyplot as plt

from mmcv.transforms import BaseTransform
from mmengine.runner.runner import Runner
from mmengine.hooks.hook import Hook
from mmengine.model.base_module import BaseModule
from mmengine.structures.base_data_element import BaseDataElement
from mmengine.registry import MODELS

from itkit.mm.mgam_models import mgam_Seg3D_Lite
from itkit.mm.mmseg_Dev3D import (EncoderDecoder_3D, Seg3DDataSample, PackSeg3DInputs,
                                  to_tensor, VolumeData, warnings)


class StatisticsData(BaseDataElement):
    """用于存储统计值（mean, std, min, max）的数据结构。"""

    def __init__(self, mean: float, std: float, min: float, max: float, count: int):
        super().__init__(stats=torch.tensor([mean, std, min, max, count]))

    @property
    def stats(self) -> Tensor:
        return self._stats

    @stats.setter
    def stats(self, value: torch.Tensor):
        assert isinstance(value, torch.Tensor), "stats 必须是一个 Tensor"
        assert value.shape == (5,), "stats Tensor 的长度必须为4"
        self.set_field(value, "_stats")

    @property
    def mean(self) -> float:
        return self._stats[0]

    @property
    def std(self) -> float:
        return self._stats[1]

    @property
    def min(self) -> float:
        return self._stats[2]

    @property
    def max(self) -> float:
        return self._stats[3]

    @property
    def count(self) -> int:
        return self._stats[4]


class Seg3DDataSample_WithStat(Seg3DDataSample):
    """A class that stores a mapping between label indices and StatisticsData objects."""

    @property
    def stat(self) -> dict[str, StatisticsData]:
        return self._stat

    @stat.setter
    def stat(self, value: dict[str, StatisticsData]) -> None:
        self.set_field(value, "_stat")

    @stat.deleter
    def stat(self) -> None:
        del self._stat

    @property
    def unique(self) -> dict[str, StatisticsData]:
        return self._unique

    @unique.setter
    def unique(self, value: dict[str, StatisticsData]) -> None:
        self.set_field(value, "_unique")

    @unique.deleter
    def unique(self) -> None:
        del self._unique


class ParseLabelDistribution(BaseTransform):
    """Support window extractor, will add a label pixel value distribution of one sample.

    Required keys:

    - img
    - gt_seg_map

    added Keys:

    - label_distr

    """

    def transform(self, results: dict):
        distr = {}
        img = results["img"]
        label = results["gt_seg_map"]

        # calculate the distribution of each label
        for idx, count in zip(*np.unique(label, return_counts=True)):
            v: np.ndarray = img[label == idx]
            distr[idx] = StatisticsData(
                mean=v.mean(), std=v.std(), min=v.min(), max=v.max(), count=count
            )
        # sort according to mean
        results["label_distr"] = OrderedDict(
            sorted(distr.items(), key=lambda x: x[1].mean)
        )

        return results


class PackSeg3DInputs_AutoWindow(PackSeg3DInputs):
    def transform(self, results: dict) -> dict:
        """Method to pack the input data for 3D segmentation.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`Seg3DDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if "img" in results:
            img = results["img"]
            if len(img.shape) < 4:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(3, 0, 1, 2)))
            else:
                img = img.transpose(3, 0, 1, 2)
                img = to_tensor(img).contiguous()
            packed_results["inputs"] = img

        data_sample = Seg3DDataSample_WithStat()

        if "gt_seg_map" in results:
            if len(results["gt_seg_map"].shape) == 3:
                data = to_tensor(results["gt_seg_map"][None].astype(np.uint8))
            else:
                warnings.warn(
                    "Please pay attention your ground truth "
                    "segmentation map, usually the segmentation "
                    "map is 3D, but got "
                    f'{results["gt_seg_map"].shape}'
                )
                data = to_tensor(results["gt_seg_map"].astype(np.uint8))
            data_sample.gt_sem_seg = VolumeData(data=data)  # type: ignore

        if "gt_seg_map_one_hot" in results:
            assert len(results["gt_seg_map_one_hot"].shape) == 4, (
                f"The shape of gt_seg_map_one_hot should be `[X, Y, Z, Classes]`, "
                f'but got {results["gt_seg_map_one_hot"].shape}'
            )
            data = to_tensor(results["gt_seg_map_one_hot"].astype(np.uint8))
            data_sample.gt_sem_seg_one_hot = VolumeData(data=data)

        # NOTE AutoWindow Requires
        if "label_distr" in results:
            data_sample.stat = results["label_distr"]

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results["data_samples"] = data_sample

        return packed_results


class DynamicParam(BaseModule):
    def __init__(
        self,
        init_method: Callable,
        num_param: int = 1,
        ensure_sign: str | None = None,
        eps: float = 0.1,
    ):
        super().__init__()
        if ensure_sign is not None:
            assert ensure_sign in ["pos", "neg"]
        self.ensure_sign = ensure_sign
        self.param = torch.nn.Parameter(torch.empty(num_param))
        init_method(self.param)
        self.eps = eps

    def forward(self, *args, **kwargs) -> Tensor:
        if self.ensure_sign == "pos":
            return torch.relu(self.param) + self.eps
        elif self.ensure_sign == "neg":
            return -torch.relu(self.param) - self.eps
        else:
            return self.param

    def status(self):
        return self.param.detach().cpu().numpy()

    def __getattr__(self, name):
        if name == "device":
            return self.param.device
        else:
            return super().__getattr__(name)


class SupportLrMultModule(BaseModule):
    def __init__(self, lr_mult: float | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_mult = lr_mult
        if lr_mult is not None:
            self.register_full_backward_hook(self.scale_gradients)

    @staticmethod
    def scale_gradients(module, grad_input, grad_output):
        return tuple(g * module.lr_mult if g is not None else g for g in grad_input)


class WindowExtractor(SupportLrMultModule):
    "Extract values from raw array using a learnable window."

    def __init__(
        self,
        in_channels: int,
        focus_range: list | tuple,
        value_range: list | tuple,
        lr_scale: float | None = None,
        momentum: float = 0.99,
        eps: float = 0.1,
        dim="3d",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.focus_range = focus_range
        self.value_range = value_range
        self.lr_scale = lr_scale
        self.momentum = momentum
        self.eps = eps
        self.dim = dim
        self.log_count = 0

        self.window_sample = torch.arange(*self.value_range)
        self.relative_focus_range = [(i - value_range[0]) / (value_range[1] - value_range[0])
                                      for i in focus_range]

        # Dynamic Weak Response - a
        self.d_wr = DynamicParam(partial(torch.nn.init.normal_, mean=1, std=0.1), ensure_sign="pos")
        # Dynamic Intense Response - b
        self.d_ir = DynamicParam(partial(torch.nn.init.normal_, mean=1, std=0.1), ensure_sign="pos")
        # Dynamic Perception Field - d
        self.d_pf = DynamicParam(partial(torch.nn.init.normal_, mean=1, std=0.1), ensure_sign=None)
        # Dynamic Range - g
        self.d_r = DynamicParam(torch.nn.init.zeros_, ensure_sign=None)
        # Dynamic Range Auxiliary Parameter
        self.d_r_a = DynamicParam(torch.nn.init.zeros_, ensure_sign="pos")
        # Global Offset - k
        self.g_o = DynamicParam(torch.nn.init.zeros_, ensure_sign=None)
        # Range Rectification Coefficient - h
        self.rrc = focus_range[1] - focus_range[0]
        # Major dynamic range for this window to handle.
        self.major_handle = (focus_range[1] + focus_range[0]) / 2
        assert self.rrc > 0
        
        self.register_buffer("_momentum_mean", torch.zeros(1, device=self.g_o.device))
        self.register_buffer("_momentum_std", torch.zeros(1, device=self.g_o.device))

    @torch.inference_mode()
    def current_response(self):
        self.window_sample = self.window_sample.to(self.g_o.device)
        response = self.forward(self.window_sample).cpu().numpy()
        return response

    @torch.inference_mode()
    def status(self) -> dict:
        return {
            "d_wr": self.d_wr.status(),
            "d_ir": self.d_ir.status(),
            "d_pf": self.d_pf.status(),
            "d_r": self.d_r.status(),
            "d_r_a": self.d_r_a.status(),
            "g_o": self.g_o.status(),
            "rrc": self.rrc,
        }

    @torch.no_grad()
    def _focus_range_momentum(self, v1: float, v2: float):
        # v1: mean, v2: std
        device = self._momentum_mean.device
        v1 = torch.tensor(v1, device=device, dtype=self._momentum_mean.dtype)
        v2 = torch.tensor(v2, device=device, dtype=self._momentum_std.dtype)
        self._momentum_mean = self._momentum_mean * self.momentum + v1 * (1 - self.momentum)
        self._momentum_std = self._momentum_std * self.momentum + v2 * (1 - self.momentum)
        return torch.stack([self._momentum_mean, self._momentum_std])

    def _focus_range_of_this_sample(self, data_samples: list[Seg3DDataSample]):
        intrs_means = []
        intrs_stds = []
        # locate interested classes' distributions according to `relative_focus_range`
        for s in data_samples:
            distr = [(class_idx, stat) for class_idx, stat in s.stat.items()]
            # len(distr) = num of classes of this sample
            intrs_classes_start = int(len(distr) * self.relative_focus_range[0])
            intrs_classes_end = int(len(distr) * self.relative_focus_range[1])
            if intrs_classes_start == intrs_classes_end:
                intrs_classes_end += 1

            intrs_classes = distr[intrs_classes_start:intrs_classes_end]
            intrs_means.extend([stat.mean for _, stat in intrs_classes])
            intrs_stds.extend([stat.std for _, stat in intrs_classes])

        batch_mean = np.array(intrs_means).mean()
        batch_std = np.array(intrs_stds).mean()

        return self._focus_range_momentum(batch_mean, batch_std)

    def loss(self, WinE_proj: Tensor, data_samples: list[Seg3DDataSample]):
        focus_target_mean, focus_target_std = self._focus_range_of_this_sample(
            data_samples
        ).to(WinE_proj.device)
        mean_loss = torch.abs(focus_target_mean / self.rrc - WinE_proj.mean())
        std_loss = torch.abs(focus_target_std / self.rrc - WinE_proj.std())
        return mean_loss + std_loss

    def forward(self, x: Tensor):
        """
        Args:
            inputs (Tensor): (...)
        """
        # Dynamic Perception Field - d
        d_pf = self.d_pf(x)
        # NOTE Dynamic Range - g
        d_r = torch.tanh(self.d_r(x)) + 1 + self.d_r_a(x) + self.eps
        # Dynamic Weak Response - a
        d_wr = self.d_wr(x) + 1 + self.eps
        # Dynamic Intense Response - b
        d_ir = self.d_ir(x) + 1 + self.eps
        # Global Offset - k
        g_o = self.g_o(x)

        exp = ((x - self.major_handle) / self.rrc + d_pf) / d_r

        response = (d_wr * torch.exp(exp) - d_ir * torch.exp(-exp)) / (
            d_wr * torch.exp(exp) + d_ir * torch.exp(-exp)
        ) + g_o

        return response


class TanhRectifier(SupportLrMultModule):
    """
    Value-Wise Projector for one window remapping operation.
    The extracted value are fine-tuned by this projector.
    """

    def __init__(
        self,
        in_channels: int,
        num_rect: int,
        momentum: float = 0.99,
        sample_nbins: int = 256,
        dim: str = "3d",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.num_rect = num_rect
        self.sample = torch.arange(sample_nbins) / sample_nbins * 10 - 5
        self.dim = dim

        self.rectify_intense = DynamicParam(
            partial(torch.nn.init.normal_, std=0.1), num_param=num_rect
        )
        self.rectify_location = DynamicParam(torch.nn.init.normal_, num_param=num_rect)

        self.momentum = momentum
        self.std_memory = torch.ones(1)

    @torch.inference_mode()
    def current_projection(self):
        self.sample = self.sample.to(device=self.rectify_intense.device)
        proj = self.forward(self.sample)
        return (self.sample.detach().cpu().numpy(), proj.cpu().numpy())

    def loss(self, inputs: Tensor, data_samples: list[Seg3DDataSample]) -> Tensor:
        """
        Limit the projector ability to ensure it's behavior,
        which aligns with the physical meaning.
        """
        with torch.no_grad():
            iter_std = inputs.std()
            if not torch.isnan(iter_std):
                self.std_memory = self.std_memory.to(device=inputs.device)
                self.std_memory = self.std_memory * self.momentum + iter_std * (1-self.momentum)

        return torch.abs(self.rectify_location().std() - self.std_memory)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs (Tensor): (...)
        """

        # NOTE Mathematical Designs
        # The adding operation between rectify_locations and inputs
        # is equivalent to the X-axis translation operation.
        rectification = (
            torch.tanh(
                self.rectify_location()  # [num_rect]
                + inputs.expand(self.num_rect, *inputs.shape).moveaxis(
                    0, -1
                )  # [..., num_rect]
            )
            * self.rectify_intense()
        )  # rectification: [..., num_rect]

        return inputs + torch.sum(rectification, dim=-1)  # [...]


class BatchCrossWindowFusion(SupportLrMultModule):
    def __init__(self, num_windows: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_fusion_weight = torch.nn.Parameter(
            torch.eye(num_windows, num_windows)
        )

    @property
    def current_fusion(self):
        return self.window_fusion_weight.detach().cpu().numpy()

    def forward(self, inputs: Tensor):
        """
        Args:
            inputs (Tensor): (Win, N, C, ...)

        Returns:
            Tensor: (N, Win*C, ...)
        """
        ori_shape = inputs.shape
        fused = torch.matmul(
            torch.softmax(self.window_fusion_weight, dim=1), inputs.flatten(1)
        ).reshape(ori_shape)
        return fused.transpose(0, 1).flatten(1, 2)  # [N, Win*C, ...]


class ParalleledMultiWindowProcessing(BaseModule):
    """The top module of Paralleled Multi-Window Processing."""

    def __init__(
        self,
        in_channels: int,
        num_windows: int = 4,
        num_rect: int = 8,
        TRec_rect_momentum: float = 0.99,
        data_range: list[int] = [-1024, 3072],
        dim="3d",
        enable_WinE_loss: bool = False,
        enable_TRec: bool = True,
        enable_TRec_loss: bool = False,
        enable_CWF: bool = True,
        lr_mult: float | None = None,
        *args, **kwargs,
    ):
        assert dim.lower() in ["2d", "3d"]
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.num_windows = num_windows
        self.num_rect = num_rect
        self.TRec_rect_momentum = TRec_rect_momentum
        self.data_range = data_range
        self.dim = dim
        self.enable_WinE_loss = enable_WinE_loss
        self.enable_TRec = enable_TRec
        self.enable_TRec_loss = enable_TRec_loss
        self.enable_CWF = enable_CWF
        self.lr_mult = lr_mult
        self._init_PMWP()

    def _init_PMWP(self):
        sub_window_ranges = self._split_range_into_milestones()

        for i in range(self.num_windows):
            setattr(
                self,
                f"window_extractor_{i}",
                WindowExtractor(
                    in_channels=self.in_channels,
                    focus_range=sub_window_ranges[i],
                    value_range=self.data_range,
                    dim=self.dim,
                    lr_mult=self.lr_mult,
                ),
            )

            if self.enable_TRec:
                setattr(
                    self,
                    f"tanh_rectifier_{i}",
                    TanhRectifier(
                        in_channels=self.in_channels,
                        num_rect=self.num_rect,
                        momentum=self.TRec_rect_momentum,
                        dim=self.dim,
                        lr_mult=self.lr_mult,
                    ),
                )

        # TODO Maybe Point-Wise Attention?
        if self.enable_CWF:
            self.cross_window_fusion = BatchCrossWindowFusion(
                self.num_windows, lr_mult=self.lr_mult
            )

    def _split_range_into_milestones(self):
        if self.num_windows <= 0:
            raise ValueError("n must be a positive integer")
        start, end = self.data_range
        n = self.num_windows

        step = (end - start) / n
        milestones = np.arange(start, end, step)
        sub_ranges = [(milestones[i], milestones[i] + max(255, step)) for i in range(n)]
        return sub_ranges

    def train_iter_info_hook(self):
        WinE = dict()
        for i in range(self.num_windows):
            for k, v in getattr(self, f"window_extractor_{i}").status().items():
                WinE[f"WinE/W{i}_{k}"] = v
        
        if self.enable_CWF:
            CrsF = self.cross_window_fusion.current_fusion
        else:
            CrsF = None

        return WinE, CrsF

    def val_epoch_info_hook(self):
        if self.enable_TRec:
            TRec = {
                f"TRec/P{i}": getattr(self, f"tanh_rectifier_{i}").current_projection()
                for i in range(self.num_windows)
            }
        else:
            TRec = None
        
        return TRec

    def feedback_losses(
        self,
        inputs: Tensor,
        features: list[Tensor],
        losses: dict,
        data_samples: list[Seg3DDataSample],
    ) -> dict:
        return losses

    @staticmethod
    def _forward_one_branch(
        WinE,
        TRec,
        inputs: Tensor,
        data_samples_for_pmwp_loss: list[Seg3DDataSample_WithStat] | None,
        num_windows: int,
        enable_WinE_loss: bool,
        enable_TRec_loss: bool,
    ):
        proj = WinE(inputs)
        if enable_WinE_loss and data_samples_for_pmwp_loss is not None:
            WinE_loss = WinE.loss(proj, data_samples_for_pmwp_loss) / num_windows
        else:
            WinE_loss = None

        if TRec is not None:
            proj = TRec.forward(proj)
        if (
            enable_TRec_loss
            and data_samples_for_pmwp_loss is not None
            and TRec is not None
        ):
            TRec_loss = TRec.loss(proj, data_samples_for_pmwp_loss) / num_windows
        else:
            TRec_loss = None

        return proj, WinE_loss, TRec_loss

    def forward(
        self,
        inputs: Tensor,
        data_samples_for_pmwp_loss=None,
    ):
        """
        Args:
            inputs (Tensor): (N, C, ...)

        Returns:
            Tensor: (N, Win*C, ...)
            PMWP Losses: dict
        """
        x = []
        pmwp_losses = {}

        for i in range(self.num_windows):
            WinE = getattr(self, f"window_extractor_{i}")
            if self.enable_TRec:
                TRec = getattr(self, f"tanh_rectifier_{i}")
            else:
                TRec = None

            proj, WinE_loss, TRec_loss = self._forward_one_branch(
                WinE,
                TRec,
                inputs,
                data_samples_for_pmwp_loss,
                self.num_windows,
                self.enable_WinE_loss,
                self.enable_TRec_loss,
            )

            if WinE_loss is not None:
                pmwp_losses[f"loss/WinE_{i}"] = WinE_loss
            if TRec_loss is not None:
                pmwp_losses[f"loss/TRec_{i}"] = TRec_loss
            x.append(proj)  # [N, C, ...]

        x = torch.stack(x)  # [Window, N, C, ...]

        if self.enable_CWF:
            x = self.cross_window_fusion(x)  # [N, Win*C, ...]
        else:
            x = x.transpose(0, 1).flatten(1, 2)  # [N, Win*C, ...]

        return x, pmwp_losses  # [N, Win*C, ...]


class AutoWindowSetting(EncoderDecoder_3D):
    "Compatible Plugin for Auto Window Setting."

    def __init__(self, pmwp: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pmwp: ParalleledMultiWindowProcessing = MODELS.build(pmwp)

    def extract_feat(self, inputs: Tensor, data_samples_for_pmwp_loss=None):
        # inputs: [N, C, ...]
        # pmwp_out: [N, num_window * C, ...]
        # TODO Downsampling Channel?
        pmwp_out, pmwp_loss = self.pmwp(inputs, data_samples_for_pmwp_loss)

        # when `data_samples_for_pmwp_loss` is not None,
        # `pmwp_loss` should not be None.
        if data_samples_for_pmwp_loss is not None:
            return super().extract_feat(pmwp_out), pmwp_loss
        else:
            return super().extract_feat(pmwp_out)

    def loss(self, inputs: Tensor, data_samples: list[Seg3DDataSample]):
        x, pmwp_loss = self.extract_feat(inputs, data_samples)
        x: list[Tensor]
        losses: dict = pmwp_loss

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        losses = self.pmwp.feedback_losses(inputs, x, losses, data_samples)

        return losses


class AutoWindowStatusLoggerHook(Hook):
    def __init__(self, dpi: int = 100, interval:int = 100):
        self.dpi = dpi
        self.interval = interval

    @staticmethod
    def iter(runner):
        if isinstance(runner._train_loop, dict) or runner._train_loop is None:
            return 0
        else:
            return runner.iter

    @torch.no_grad()
    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch = None,
                         outputs = None) -> None:
        current_iter = self.iter(runner)
        if current_iter % self.interval != 0:
            return
        
        model: ParalleledMultiWindowProcessing = runner.model.pmwp
        WinE, CrsF = model.train_iter_info_hook()
        
        if not hasattr(self, 'WinE_Figure'):
            self.WinE_Figure = plt.figure(figsize=(4, 3))
        for k in list(WinE.keys()):
            if "RespMat" in k:
                signal, response = WinE.pop(k)
                buf = BytesIO()
                self.WinE_Figure.clf()
                self.WinE_Figure.gca().plot(signal, response)
                self.WinE_Figure.tight_layout()
                self.WinE_Figure.savefig(buf, format="png", dpi=self.dpi)
                image = np.array(Image.open(buf).convert("RGB"))
                runner.visualizer.add_image(k, image, current_iter)

        runner.visualizer.add_scalars(
            WinE,
            step=current_iter,
            file_path=f"{runner.timestamp}-AutoWindowStatus.json",
        )

        if CrsF is not None:
            if not hasattr(self, 'CrsF_Figure'):
                self.CrsF_Figure = plt.figure(figsize=(4, 3))
            self.CrsF_Figure.clf()
            buf = BytesIO()
            seaborn.heatmap(CrsF, cmap="rainbow", ax=self.CrsF_Figure.gca())
            self.CrsF_Figure.savefig(buf, format="png")
            image = np.array(Image.open(buf).convert("RGB"))
            runner.visualizer.add_image("CrsF", image, current_iter)

    @torch.no_grad()
    def before_val_epoch(self, runner: Runner) -> None:
        model: ParalleledMultiWindowProcessing = runner.model.pmwp
        TRec = model.val_epoch_info_hook()
        
        if TRec is not None:
            if not hasattr(self, 'TRec_Figure'):
                self.TRec_Figure = plt.figure(figsize=(4, 3))
            current_iter = self.iter(runner)
            
            for name, proj in TRec.items():
                self.TRec_Figure.clf()
                self.TRec_Figure.gca().plot(proj[0], proj[1])
                self.TRec_Figure.gca().plot([0, 1], [0, 1], transform=plt.gca().transAxes)
                self.TRec_Figure.tight_layout()
                buf = BytesIO()
                self.TRec_Figure.savefig(buf, format="png", dpi=self.dpi)
                image = np.array(Image.open(buf).convert("RGB"))
                runner.visualizer.add_image(name, image, current_iter)


class AutoWindowLite(mgam_Seg3D_Lite):
    """New framrwork compatible with new lite mgam framework"""
    def __init__(self, 
                 pmwp:dict|None=None, 
                 inference_EmptyCache:bool=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_EmptyCache = inference_EmptyCache
        self.pmwp:ParalleledMultiWindowProcessing = MODELS.build(pmwp) if pmwp is not None else None

    @torch.no_grad()
    def get_param(self, inputs:Tensor):
        # inputs: [N, C, Z, Y, X]
        
        from torchinfo import summary
        pmwp_param = summary(
            self.pmwp,
            input_size=(1, *inputs.shape[-3:]),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            row_settings=["var_names"],
            dtypes=[torch.float16],
            col_width=15,
        )
        backbone_param = summary(
            self.backbone,
            input_size=(1, self.pmwp.num_windows, *inputs.shape[-3:]),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            row_settings=["var_names"],
            dtypes=[torch.float16],
            col_width=15,
        )
        
        pdb.set_trace()

    def loss(self, inputs: Tensor, data_samples: list[Seg3DDataSample]):
        # self.get_param(inputs)
        
        losses = {}
        if self.pmwp is not None:
            inputs, pmwp_losses = self.pmwp(inputs, data_samples)
            losses.update(pmwp_losses)
        main_losses = super().loss(inputs, data_samples)
        losses.update(main_losses)
        return losses

    def predict(self, inputs:Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Sequence[BaseDataElement]:
        if self.pmwp is not None:
            inputs, pmwp_losses = self.pmwp(inputs, data_samples)
        result = super().predict(inputs, data_samples)
        if self.inference_EmptyCache:
            torch.cuda.empty_cache()
        return result
