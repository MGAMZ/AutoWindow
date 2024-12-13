import os
import pdb
import math

from cv2 import exp
import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix

from inference import Inferencer_3D
from mgamdata.models.AutoWindow import AutoWindowSetting
from mgamdata.dataset.AbdomenCT_1K.meta import CLASS_INDEX_MAP  # {name: index}


CMAP_COLOR = ["#50b2cd", "#c490c4"]
CMAP_SEQ_COLOR = ["GnBu", "RdPu"]


def draw_combined_WinE_resp(ori: np.ndarray, resps: np.ndarray, save_path: str):
    num_resps = len(resps)
    cols = num_resps + 1  # 包含原始图像
    rows = 2  # 第一行为图像，第二行为直方图

    fig, axes = plt.subplots(
        rows, cols, figsize=(3.5 * cols, 4), gridspec_kw={"height_ratios": [1, 0.2]}
    )

    # 绘制原始图像
    img_ori = ori[len(ori) // 2]
    ax_img_ori = axes[0, 0]
    img_display = ax_img_ori.imshow(img_ori, cmap="gray")
    ax_img_ori.set_title("Original")
    ax_img_ori.axis("off")
    fig.colorbar(img_display, ax=ax_img_ori, orientation="vertical")

    # 绘制原始直方图
    ax_hist_ori = axes[1, 0]
    ax_hist_ori.hist(ori.flatten(), bins=100, color=CMAP_COLOR[1], alpha=0.7)
    ax_hist_ori.set_xlabel("Hounsfield Units")
    ax_hist_ori.set_ylabel("Frequency")
    ax_hist_ori.set_yscale("symlog")
    ax_hist_ori.set_ylim(10e0, 10e6)
    ax_hist_ori.yaxis.set_major_locator(MaxNLocator(nbins=3))

    # 绘制响应图像和直方图
    for i, resp in enumerate(resps):
        # 图像
        img = resp[len(resp) // 2]
        ax_img = axes[0, i + 1]
        img_display = ax_img.imshow(img, cmap="gray")
        ax_img.set_title(f"Auto Window {i+1}")
        ax_img.axis("off")
        fig.colorbar(img_display, ax=ax_img, orientation="vertical")

        # 直方图
        ax_hist = axes[1, i + 1]
        ax_hist.hist(resp.flatten(), bins=100, color=CMAP_COLOR[0], alpha=0.7)
        ax_hist.set_xlabel("Response")
        ax_hist.set_xlim(-1.8, 1.0)
        ax_hist.set_ylim(10e2, 10e6)
        ax_hist.set_yscale("symlog")
        ax_hist.yaxis.set_major_locator(MaxNLocator(nbins=3))

    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=300,
    )
    plt.close(fig)


def draw_crsf_resp(
    inputs: np.ndarray, crsf_resp: np.ndarray, num_windows: int, save_path: str
):
    crsf_resp = np.stack(np.split(crsf_resp, num_windows, axis=0)).squeeze()
    inputs, crsf_resp = np.flip(inputs, axis=2), np.flip(crsf_resp, axis=2)
    inputs = inputs[:, inputs.shape[1] // 2]  # [win, Y, X]
    crsf_resp = crsf_resp[:, crsf_resp.shape[1] // 2]  # [win, Y, X]

    vminmax = np.stack([inputs, crsf_resp], axis=1)
    vmin = vminmax.min(axis=(1, 2, 3))
    vmax = vminmax.max(axis=(1, 2, 3))

    fig, axes = plt.subplots(2, num_windows, figsize=(3 * num_windows, 4))

    for i in range(num_windows):
        # 绘制 inputs
        ax_input = axes[0, i]
        ax_input.hist(inputs[i].flatten(), bins=200, color=CMAP_COLOR[1], alpha=0.7)
        ax_input.set_title(f"Auto Window {i + 1}")
        # ax_input.set_xlim(vmin[i], vmax[i])
        ax_input.set_yscale("log")
        ax_input.tick_params(axis="x", rotation=30)
        if i == 0:
            ax_input.set_ylabel("Tanh-Rectifier Output")

        # 绘制 crsf_resp
        ax_resp = axes[1, i]
        ax_resp.hist(crsf_resp[i].flatten(), bins=200, color=CMAP_COLOR[0], alpha=0.7)
        # ax_resp.set_xlim(vmin[i], vmax[i])
        ax_resp.set_yscale("log")
        ax_resp.tick_params(axis="x", rotation=30)
        if i == 0:
            ax_resp.set_ylabel("Cross-Window Fused")

    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=300,
    )
    plt.close(fig)


def draw_confusion_matrix(gt: np.ndarray, pred: np.ndarray, save_path: str):
    """
    Args:
        gt (np.ndarray): size (Z, Y, X)
        pred (np.ndarray): size (Z, Y, X)
    """
    # 计算混淆矩阵
    cm = confusion_matrix(gt.flatten(), pred.flatten())[1:]
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(6, 4.5))
    # 绘制混淆矩阵，应用对数归一化
    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap=CMAP_SEQ_COLOR[0],
        norm=LogNorm(vmin=1000, vmax=30000, clip=True),
        alpha=0.7,
    )
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count (Log Scale)", rotation=270, labelpad=15)
    # 设置标题和标签
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    # 设置刻度和标签
    ticks = [
        {v: k for k, v in CLASS_INDEX_MAP.items()}[idx]
        for idx in np.union1d(np.unique(gt), np.unique(pred))
    ]
    ax.set_xticks(np.arange(len(ticks)))
    ax.set_yticks(np.arange(len(ticks) - 1))
    ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.set_yticklabels(ticks[1:])
    # 添加数值标签
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    # 调整布局并保存图像
    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=500,
    )
    plt.close(fig)


def analyze_one_exp(config_path, ckpt_path, itk_image_path, inst_norm: bool, manual_win: bool, invert_y:bool=False):
    exp_name = os.path.basename(os.path.dirname(config_path))
    itk_mask_path = itk_image_path.replace("image", "label")

    itk_image = sitk.ReadImage(itk_image_path)
    gt_image = sitk.ReadImage(itk_mask_path)
    image_array = sitk.GetArrayFromImage(itk_image)
    gt_array = sitk.GetArrayFromImage(gt_image)

    assert (inst_norm and manual_win) is not True
    if inst_norm:
        image_array = (image_array - image_array.min()) / image_array.std()
    if manual_win:
        ww = 400
        wl = 40
        image_array = np.clip(image_array, wl - ww // 2, wl + ww // 2)
        image_array = (image_array - (wl - ww // 2)) / ww

    inferencer = Inferencer_3D(config_path, ckpt_path)
    model: AutoWindowSetting = inferencer.model

    with torch.inference_mode():
        wine_resps = []
        tanh_resps = []
        if hasattr(model, "pmwp"):
            for i in range(model.pmwp.num_windows):
                wine = eval(f"model.pmwp.window_extractor_{i}")
                resp = wine.forward(torch.from_numpy(image_array).cuda())
                wine_resps.append(resp.cpu().numpy())
                tanh = eval(f"model.pmwp.tanh_rectifier_{i}")
                resp = tanh.forward(resp)
                tanh_resps.append(resp.cpu().numpy())

            num_windows = len(tanh_resps)
            all_window = torch.from_numpy(np.stack(tanh_resps)).cuda().float()
            crsf_resp = (
                model.pmwp.cross_window_fusion(all_window[:, None])
                .cpu()
                .numpy()
                .squeeze()
            )

    pred_array = inferencer.Inference_FromNDArray(image_array)
    pred_array = pred_array.argmax(dim=0).cpu().numpy()
    print("Inference Done.")

    wine_resps = np.stack(wine_resps) if len(wine_resps) > 0 else None
    if invert_y:
        image_array = image_array[..., ::-1, :]
        wine_resps = wine_resps[..., ::-1, :] if wine_resps is not None else None
    if wine_resps is not None:
        draw_combined_WinE_resp(
            image_array,
            wine_resps,
            f"/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures/combined_wine/combined_wine_{exp_name}.png",
        )
        draw_crsf_resp(
            all_window.cpu().numpy(),
            crsf_resp,
            num_windows,
            f"/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures/crsf_resp/crsf_resp_{exp_name}.png",
        )
    draw_confusion_matrix(
        gt_array,
        pred_array,
        f"/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures/cm/cm_{exp_name}.png",
    )

    print("Drown.")


if __name__ == "__main__":
    # AbdomenCT_1K
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.6.0.InstanceNorm_AbdomenCT_1K/MedNeXt.py", 
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.6.0.InstanceNorm_AbdomenCT_1K/MedNeXt/iter_200000.pth", 
        itk_image_path="/file1/mgam_datasets/AbdomenCT_1K/spacing2_mha/image/01062.mha",
        inst_norm=True,
        manual_win=False,
        invert_y=True,
    )
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.6.2.Window4/MedNeXt.py", 
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.6.2.Window4/MedNeXt/best_Perf_mDice_iter_155000.pth", 
        itk_image_path="/file1/mgam_datasets/AbdomenCT_1K/spacing2_mha/image/01062.mha",
        inst_norm=False,
        manual_win=False,
        invert_y=True,
    )
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.6.3.Window8/MedNeXt.py", 
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.6.3.Window8/MedNeXt/best_Perf_mDice_iter_175000.pth", 
        itk_image_path="/file1/mgam_datasets/AbdomenCT_1K/spacing2_mha/image/01062.mha",
        inst_norm=False,
        manual_win=False,
        invert_y=True,
    )
    
    # KiTS23
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.5.0.InstanceNorm_kits23/MedNeXt.py", 
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.5.0.InstanceNorm_kits23/MedNeXt/iter_200000.pth", 
        itk_image_path="/file1/mgam_datasets/KiTS23/spacing2_mha/image/00523.mha",
        inst_norm=True,
        manual_win=False,
    )
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.5.2.Window4/MedNeXt.py", 
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.5.2.Window4/MedNeXt/best_Perf_mDice_iter_175000.pth", 
        itk_image_path="/file1/mgam_datasets/KiTS23/spacing2_mha/image/00523.mha",
        inst_norm=False,
        manual_win=False,
    )
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.5.3.Window8/MedNeXt.py", 
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.5.3.Window8/MedNeXt/best_Perf_mDice_iter_90000.pth", 
        itk_image_path="/file1/mgam_datasets/KiTS23/spacing2_mha/image/00523.mha",
        inst_norm=False,
        manual_win=False,
    )

    # ImageTBAD
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.3.3.InstanceNorm/MedNeXt.py", 
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.3.3.InstanceNorm/MedNeXt/iter_200000.pth", 
        itk_image_path="/file1/mgam_datasets/ImageTBAD/spacing2_mha/image/180.mha",
        inst_norm=True,
        manual_win=False,
    )
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.3.0.Window4_ImageTBAD/MedNeXt.py", 
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.3.0.Window4_ImageTBAD/MedNeXt/best_Perf_mDice_iter_135000.pth", 
        itk_image_path="/file1/mgam_datasets/ImageTBAD/spacing2_mha/image/180.mha",
        inst_norm=False,
        manual_win=False,
    )
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.3.1.Window8/MedNeXt.py", 
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.3.1.Window8/MedNeXt/best_Perf_mDice_iter_130000.pth", 
        itk_image_path="/file1/mgam_datasets/ImageTBAD/spacing2_mha/image/180.mha",
        inst_norm=False,
        manual_win=False,
    )