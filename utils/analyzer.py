import os
import pdb
from re import A
from termios import VMIN

import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LogNorm, Normalize

from inference import Inferencer_3D
from mgamdata.models.AutoWindow import AutoWindowSetting
from mgamdata.dataset.AbdomenCT_1K.meta import CLASS_INDEX_MAP  # {name: index}


def draw_WinE_resp(resps: list[np.ndarray]):
    fig, axes = plt.subplots(2, 4, figsize=(10, 6))
    for i, ax in enumerate(axes.flat):
        img = resps[i][len(resps[i]) // 2]  # select the middle slice
        ax.set_title(f"Window {i+1}")
        cbar = fig.colorbar(
            ax.imshow(img, cmap="gray"), ax=ax, orientation="horizontal"
        )

        num_ticks = 6
        x_ticks = np.linspace(0, img.shape[1], num=num_ticks)
        y_ticks = np.linspace(0, img.shape[0], num=num_ticks)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f"{int(t*2)}" for t in x_ticks])
        ax.set_yticklabels([f"{int(t*2)}" for t in y_ticks])
        ax.set_ylabel("mm")

    fig.tight_layout()
    fig.savefig(
        "/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures/temp_wine.png",
        dpi=300,
    )


def draw_WinE_resp_distr(ori: np.ndarray, resps: list[np.ndarray]):
    import matplotlib.pyplot as plt
    import math

    num_resps = len(resps)
    cols = 4
    rows = math.ceil(num_resps / cols)

    # 创建一个包含ori和resps的子图布局
    fig = plt.figure(figsize=(10, 1.7 * (rows + 1)))
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1] + [1] * rows)

    # 绘制原始图像的 HU 分布，跨两列
    ax_ori = fig.add_subplot(gs[0, :])
    ax_ori.hist(ori.flatten(), bins=100, color="blue", alpha=0.7)
    ax_ori.set_title("Original")
    ax_ori.set_xlabel("Hounsfield Units")
    ax_ori.set_ylabel("Frequency")
    ax_ori.set_yscale("symlog")
    ax_ori.set_ylim(10e0, 10e6)

    # 绘制每个响应的 HU 分布，按照两列排列
    for idx, resp in enumerate(resps):
        row = (idx // cols) + 1
        col = idx % cols
        ax = fig.add_subplot(gs[row, col])
        ax.hist(resp.flatten(), bins=100, color="green", alpha=0.7)
        ax.set_title(f"Window {idx + 1}")
        if row == rows:
            ax.set_xlabel("Response")
        if col == 0:
            ax.set_ylabel("Frequency")
        ax.set_xlim(-1.8, 1.0)
        ax.set_ylim(10e2, 10e6)
        ax.set_yscale("symlog")

    fig.tight_layout()
    fig.savefig(
        "/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures/temp_wine_distr.png",
        dpi=300,
    )
    plt.close(fig)


def draw_crsf_resp(inputs: np.ndarray, crsf_resp: np.ndarray, num_windows: int):
    crsf_resp = np.stack(np.split(crsf_resp, num_windows, axis=0)).squeeze()
    inputs, crsf_resp = np.flip(inputs, axis=2), np.flip(crsf_resp, axis=2)
    inputs = inputs[:, inputs.shape[1] // 2]            # [win, Y, X]
    crsf_resp = crsf_resp[:, crsf_resp.shape[1] // 2]   # [win, Y, X]

    vminmax = np.stack([inputs, crsf_resp], axis=1)
    vmin = vminmax.min(axis=(1,2,3))
    vmax = vminmax.max(axis=(1,2,3))

    fig, axes = plt.subplots(2, num_windows, figsize=(3 * num_windows, 6))

    for i in range(num_windows):
        # 绘制 inputs
        ax_input = axes[0, i]
        ax_input.hist(inputs[i].flatten(), bins=200)
        ax_input.set_title(f"Window {i + 1}")
        # ax_input.set_xlim(vmin[i], vmax[i])
        ax_input.set_yscale('log')
        ax_input.tick_params(axis='x', rotation=30)
        if i == 0:
            ax_input.set_ylabel("Tanh-Rectifier Output")

        # 绘制 crsf_resp
        ax_resp = axes[1, i]
        ax_resp.hist(crsf_resp[i].flatten(), bins=200)
        # ax_resp.set_xlim(vmin[i], vmax[i])
        ax_resp.set_yscale('log')
        ax_resp.tick_params(axis='x', rotation=30)
        if i == 0:
            ax_resp.set_ylabel("Cross-Window Fused")

    fig.tight_layout()
    fig.savefig(
        "/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures/crsf_resp.png",
        dpi=300,
    )
    plt.close(fig)


def draw_confusion_matrix(gt: np.ndarray, pred: np.ndarray):
    """
    Args:
        gt (np.ndarray): size (Z, Y, X)
        pred (np.ndarray): size (Z, Y, X)
    """
    from sklearn.metrics import confusion_matrix

    # 计算混淆矩阵
    pred_labels = list(CLASS_INDEX_MAP.keys())
    truth_labels = pred_labels[1:]
    cm = confusion_matrix(gt.flatten(), pred.flatten())[1:]
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(5, 4))
    # 绘制混淆矩阵，应用对数归一化
    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Wistia",
        norm=LogNorm(vmin=1000, vmax=30000, clip=True)
    )
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count (Log Scale)", rotation=270, labelpad=15)
    # 设置标题和标签
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    # 设置刻度和标签
    ax.set_xticks(np.arange(len(pred_labels)))
    ax.set_yticks(np.arange(len(truth_labels)))
    ax.set_xticklabels(pred_labels, rotation=45, ha="right")
    ax.set_yticklabels(truth_labels)
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
        "/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures/temp_cm.png",
        dpi=500,
    )
    plt.close(fig)


if __name__ == "__main__":
    config_path = "/file1/mgam_projects/AutoWindow/configs/0.0.3.1.Window8/MedNeXt.py"
    ckpt_path = "/file1/mgam_projects/AutoWindow/work_dirs/0.0.3.1.Window8/MedNeXt/iter_200000.pth"
    itk_image_path = "/file1/mgam_datasets/ImageTBAD/spacing2_mha/image/192.mha"
    itk_mask_path = itk_image_path.replace("image", "label")

    itk_image = sitk.ReadImage(itk_image_path)
    gt_image = sitk.ReadImage(itk_mask_path)
    image_array = sitk.GetArrayFromImage(itk_image)
    gt_array = sitk.GetArrayFromImage(gt_image)

    # Optional: Normalize the image
    # image_array = (image_array - image_array.min()) / image_array.std()
    
    # Optional: Manual Window Setting
    ww = 400
    wl = 40
    image_array = np.clip(image_array, wl-ww//2, wl+ww//2)
    image_array = (image_array - (wl-ww//2)) / ww

    inferencer = Inferencer_3D(config_path, ckpt_path)
    model: AutoWindowSetting = inferencer.model

    with torch.inference_mode():
        wine_resps = []
        tanh_resps = []
        if hasattr(model, "pmwp"):
            for i in range(8):
                wine = eval(f"model.pmwp.window_extractor_{i}")
                resp = wine.forward(torch.from_numpy(image_array).cuda())
                wine_resps.append(resp.cpu().numpy())
                tanh = eval(f"model.pmwp.tanh_rectifier_{i}")
                resp = tanh.forward(resp)
                tanh_resps.append(resp.cpu().numpy())
        
        num_windows = len(tanh_resps)
        all_window = torch.from_numpy(np.stack(tanh_resps)).cuda().float()
        crsf_resp = model.pmwp.cross_window_fusion(all_window[:, None]).cpu().numpy().squeeze()
    
    pred_array = inferencer.Inference_FromNDArray(image_array)
    pred_array = pred_array.argmax(dim=0).cpu().numpy()
    print("Inference Done.")
    
    if wine_resps:
        # draw_WinE_resp(wine_resps)
        # draw_WinE_resp_distr(image_array, wine_resps)
        draw_crsf_resp(all_window.cpu().numpy(), 
                       crsf_resp, 
                       num_windows)
    # draw_confusion_matrix(gt_array, pred_array)

    print("Drown.")
