import os
import pdb
import math
from tkinter import Image
from tqdm import tqdm

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


def draw_confusion_matrix(
    gt: np.ndarray, pred: np.ndarray, class_idx_map: dict, save_path: str
):
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
        {v: k for k, v in class_idx_map.items()}[idx]
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


def draw_HU_scatter(
    image_array: np.ndarray,
    gt_array: np.ndarray,
    pred_array: np.ndarray,
    sub_win: np.ndarray,
    class_index_map: dict,
    save_path: str,
):
    plt.figure(figsize=(10, 6))

    # 获取所有预测类别
    unique_classes = np.unique(pred_array)
    # 创建颜色映射
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    class_color_map = dict(zip(unique_classes, colors))

    # 对每个sub_win绘制散点
    for i in range(sub_win.shape[0]):
        # 获取当前sub_win的值和对应位置的预测标签
        current_values = sub_win[i]
        current_preds = pred_array

        # 对每个类别分别绘制
        for cls in tqdm(
            unique_classes,
            desc=f"Drawing HU Scatter for Sub-Win{i+1}",
            leave=False,
            dynamic_ncols=True,
        ):
            mask = current_preds == cls
            if mask.any():  # 如果存在该类别的点
                values = current_values[mask]
                x_coords = np.full_like(values, i)
                plt.scatter(
                    x_coords,
                    values,
                    c=[class_color_map[cls]],
                    alpha=0.5,
                    s=1,
                    label=f"Class {cls}" if i == 0 else None,
                )  # 只在第一个window添加图例

    plt.xlabel("Sub-window Index")
    plt.ylabel("HU Values")
    plt.title("HU Values Distribution in Different Sub-windows")
    plt.grid(True)

    # 添加图例，并调整位置避免遮挡
    if len(unique_classes) > 1:  # 只有多个类别时才添加图例
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # 保存图像时确保完整保存，包括图例
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def draw_HU_violinplot(
    image_array: np.ndarray,
    gt_array: np.ndarray,
    pred_array: np.ndarray,
    sub_win: np.ndarray,
    class_index_map: dict,
    save_path: str,
):
    num_subwins = sub_win.shape[0]
    unique_classes = sorted(np.unique(pred_array))
    colors = plt.cm.GnBu(np.linspace(0, 1, len(unique_classes)))
    
    # 创建子图
    fig, axes = plt.subplots(1, num_subwins, figsize=(5*num_subwins, 6), sharey=False)
    
    # 确保axes是数组，即使只有一个子图
    if num_subwins == 1:
        axes = [axes]
    
    # 为每个window创建单独的小提琴图
    for i in range(num_subwins):
        current_values = sub_win[i]
        plot_data = []
        positions = []
        
        # 收集每个类别的数据
        for j, cls in enumerate(unique_classes):
            mask = pred_array == cls
            if mask.any():
                values = current_values[mask]
                plot_data.append(values)
                positions.append(j)
        
        # 在对应的子图上绘制小提琴图
        parts = axes[i].violinplot(plot_data, positions=positions, widths=0.8)
        
        # 设置每个violin的颜色
        for j, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[j])
            pc.set_alpha(0.7)
        
        # 设置子图的标题和标签
        axes[i].set_title(f'Auto Window {i}')
        axes[i].set_xticks(range(len(unique_classes)))
        axes[i].set_xticklabels([f'Class {cls}' for cls in unique_classes], rotation=45)
        axes[i].grid(True, axis='y')
        
        # 只给最左边的子图添加y轴标签
        if i == 0:
            axes[i].set_ylabel('Response')
    
    # 添加图例到图形级别
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7) 
                      for i in range(len(unique_classes))]
    fig.legend(legend_elements, 
              [f'Class {cls}' for cls in unique_classes],
              bbox_to_anchor=(1.02, 0.5),
              loc='center left')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_one_exp(
    config_path: str,
    ckpt_path: str,
    class_idx_map: dict,
    itk_image_path: str,
    itk_mask_path: str,
    save_root: str,
    inst_norm: bool,
    manual_win: bool,
    invert_y: bool = False,
    invert_channel: bool = False,
):
    # load

    exp_name = os.path.basename(os.path.dirname(config_path))

    itk_image = sitk.ReadImage(itk_image_path)
    gt_image = sitk.ReadImage(itk_mask_path)
    image_array = sitk.GetArrayFromImage(itk_image)
    gt_array = sitk.GetArrayFromImage(gt_image)

    # calculation

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
    if invert_channel:
        image_array = image_array.transpose(2, 1, 0)
        gt_array = gt_array.transpose(2, 1, 0)
        pred_array = pred_array.transpose(2, 1, 0)
        wine_resps = (
            wine_resps.transpose(0, 3, 2, 1) if wine_resps is not None else None
        )

    if invert_y:
        image_array = image_array[..., ::-1, :]
        wine_resps = wine_resps[..., ::-1, :] if wine_resps is not None else None
        gt_array = gt_array[..., ::-1, :]
        pred_array = pred_array[..., ::-1]

    # visualization

    if wine_resps is not None:
        # draw_combined_WinE_resp(
        #     image_array,
        #     wine_resps,
        #     os.path.join(save_root, "combined_wine", f"combined_wine_{exp_name}.png"),
        # )
        # draw_crsf_resp(
        #     all_window.cpu().numpy(),
        #     crsf_resp,
        #     num_windows,
        #     os.path.join(save_root, "crsf_resp", f"crsf_resp_{exp_name}.png"),
        # )
        # draw_HU_scatter(
        #     image_array,
        #     gt_array,
        #     pred_array,
        #     wine_resps,
        #     class_idx_map,
        #     os.path.join(save_root, "HU_scatter", f"HU_scatter_{exp_name}.png"),
        # )
        draw_HU_violinplot(
            image_array,
            gt_array,
            pred_array,
            wine_resps,
            class_idx_map,
            os.path.join(save_root, "HU_scatter", f"HU_scatter_{exp_name}.png"),
        )
    # draw_confusion_matrix(
    #     gt_array,
    #     pred_array,
    #     class_idx_map,
    #     os.path.join(save_root, "cm", f"cm_{exp_name}.png"),
    # )

    print("Drown.")


if __name__ == "__main__":
    from mgamdata.dataset.AbdomenCT_1K.meta import (
        CLASS_INDEX_MAP as AbdomenCT1K_Map,
    )  # {name: index}
    from mgamdata.dataset.KiTS23.meta import CLASS_INDEX_MAP as KiTS23_Map
    from mgamdata.dataset.ImageTBAD.meta import CLASS_INDEX_MAP as ImageTBAD_Map

    # AbdomenCT_1K
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.6.0.InstanceNorm_AbdomenCT_1K/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.6.0.InstanceNorm_AbdomenCT_1K/MedNeXt/iter_200000.pth",
        class_idx_map=AbdomenCT1K_Map,
        itk_image_path="/file1/mgam_datasets/AbdomenCT_1K/spacing2_mha/image/01062.mha",
        itk_mask_path="/file1/mgam_datasets/AbdomenCT_1K/spacing2_mha/label/01062.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=True,
        manual_win=False,
        invert_y=True,
    )
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.6.2.Window4/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.6.2.Window4/MedNeXt/best_Perf_mDice_iter_155000.pth",
        class_idx_map=AbdomenCT1K_Map,
        itk_image_path="/file1/mgam_datasets/AbdomenCT_1K/spacing2_mha/image/01062.mha",
        itk_mask_path="/file1/mgam_datasets/AbdomenCT_1K/spacing2_mha/label/01062.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=False,
        manual_win=False,
        invert_y=True,
    )
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.6.3.Window8/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.6.3.Window8/MedNeXt/best_Perf_mDice_iter_175000.pth",
        class_idx_map=AbdomenCT1K_Map,
        itk_image_path="/file1/mgam_datasets/AbdomenCT_1K/spacing2_mha/image/01062.mha",
        itk_mask_path="/file1/mgam_datasets/AbdomenCT_1K/spacing2_mha/label/01062.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=False,
        manual_win=False,
        invert_y=True,
    )

    # KiTS23
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.5.0.InstanceNorm_kits23/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.5.0.InstanceNorm_kits23/MedNeXt/iter_200000.pth",
        class_idx_map=KiTS23_Map,
        itk_image_path="/file1/mgam_datasets/KiTS23/spacing2_mha/image/00523.mha",
        itk_mask_path="/file1/mgam_datasets/KiTS23/spacing2_mha/label/00523.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=True,
        manual_win=False,
    )
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.5.2.Window4/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.5.2.Window4/MedNeXt/best_Perf_mDice_iter_175000.pth",
        class_idx_map=KiTS23_Map,
        itk_image_path="/file1/mgam_datasets/KiTS23/spacing2_mha/image/00523.mha",
        itk_mask_path="/file1/mgam_datasets/KiTS23/spacing2_mha/label/00523.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=False,
        manual_win=False,
    )
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.5.3.Window8/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.5.3.Window8/MedNeXt/best_Perf_mDice_iter_90000.pth",
        class_idx_map=KiTS23_Map,
        itk_image_path="/file1/mgam_datasets/KiTS23/spacing2_mha/image/00523.mha",
        itk_mask_path="/file1/mgam_datasets/KiTS23/spacing2_mha/label/00523.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=False,
        manual_win=False,
    )

    # ImageTBAD
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.3.3.InstanceNorm/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.3.3.InstanceNorm/MedNeXt/iter_200000.pth",
        class_idx_map=ImageTBAD_Map,
        itk_image_path="/file1/mgam_datasets/ImageTBAD/spacing2_mha/image/180.mha",
        itk_mask_path="/file1/mgam_datasets/KiTS23/spacing2_mha/label/00523.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=True,
        manual_win=False,
    )
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.3.0.Window4_ImageTBAD/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.3.0.Window4_ImageTBAD/MedNeXt/best_Perf_mDice_iter_135000.pth",
        class_idx_map=ImageTBAD_Map,
        itk_image_path="/file1/mgam_datasets/ImageTBAD/spacing2_mha/image/180.mha",
        itk_mask_path="/file1/mgam_datasets/KiTS23/spacing2_mha/label/00523.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=False,
        manual_win=False,
    )
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.3.1.Window8/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.3.1.Window8/MedNeXt/best_Perf_mDice_iter_130000.pth",
        class_idx_map=ImageTBAD_Map,
        itk_image_path="/file1/mgam_datasets/ImageTBAD/spacing2_mha/image/180.mha",
        itk_mask_path="/file1/mgam_datasets/KiTS23/spacing2_mha/label/00523.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=False,
        manual_win=False,
    )
