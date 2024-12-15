import os
import pdb
from tqdm import tqdm

import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix

from inference import Inferencer_3D
from mgamdata.models.AutoWindow import AutoWindowSetting



CMAP_SEQ_COLOR = ["winter", "RdPu"]
DEFAULT_CMAP = plt.get_cmap(CMAP_SEQ_COLOR[0])
CMAP_COLOR = [DEFAULT_CMAP(32), DEFAULT_CMAP(224)]


def draw_combined_WinE_resp(ori: np.ndarray, resps: np.ndarray, save_path: str):
    num_resps = len(resps)
    cols = num_resps + 1  # 包含原始图像
    rows = 2  # 第一行为图像，第二行为直方图

    fig, axes = plt.subplots(
        rows, cols, figsize=(3.5 * cols, 4), gridspec_kw={"height_ratios": [1, 0.2]}
    )

    def get_vmin_vmax(img):
        sorted_pixels = np.sort(img.flatten())
        sorted_pixels_unique = np.sort(np.unique(img))
        vmin = sorted_pixels[int(0.05 * len(sorted_pixels))]
        vmax = sorted_pixels[int(0.995 * len(sorted_pixels))]
        return vmin, vmax

    # 绘制原始图像
    img_ori = ori[len(ori) // 2]
    vmin, vmax = get_vmin_vmax(img_ori)
    ax_img_ori = axes[0, 0]
    img_display = ax_img_ori.imshow(img_ori, cmap="gray", vmin=vmin, vmax=vmax)
    ax_img_ori.set_title("Original")
    ax_img_ori.axis("off")
    fig.colorbar(img_display, ax=ax_img_ori, orientation="vertical")

    # 绘制原始直方图
    ax_hist_ori = axes[1, 0]
    ax_hist_ori.hist(ori.flatten(), bins=100, color=CMAP_COLOR[1], alpha=0.7)
    ax_hist_ori.axvline(vmin, color="k", linestyle="dashed", linewidth=1)
    ax_hist_ori.axvline(vmax, color="k", linestyle="dashed", linewidth=1)
    ax_hist_ori.set_xlabel("Hounsfield Units")
    ax_hist_ori.set_ylabel("Frequency")
    ax_hist_ori.set_yscale("symlog")
    ax_hist_ori.set_ylim(10e0, 10e6)
    ax_hist_ori.yaxis.set_major_locator(MaxNLocator(nbins=3))

    # 绘制响应图像和直方图
    for i, resp in tqdm(
        enumerate(resps),
        desc="Drawing Combined WinE Resp",
        dynamic_ncols=True,
        leave=False,
    ):
        # 图像
        img = resp[len(resp) // 2]
        vmin, vmax = get_vmin_vmax(img)
        ax_img = axes[0, i + 1]
        img_display = ax_img.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax_img.set_title(f"Auto Window {i+1}")
        ax_img.axis("off")
        fig.colorbar(img_display, ax=ax_img, orientation="vertical")

        # 直方图
        ax_hist = axes[1, i + 1]
        ax_hist.hist(resp.flatten(), bins=100, color=CMAP_COLOR[0], alpha=0.7)
        ax_hist.axvline(vmin, color="k", linestyle="dashed", linewidth=1)
        ax_hist.axvline(vmax, color="k", linestyle="dashed", linewidth=1)
        ax_hist.set_xlabel("Response")
        ax_hist.set_xlim(-1.8, 1.0)
        ax_hist.set_ylim(10e2, 10e6)
        ax_hist.set_yscale("symlog")
        ax_hist.yaxis.set_major_locator(MaxNLocator(nbins=3))

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
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

    for i in tqdm(
        range(num_windows), desc="Drawing CrsF Resp", dynamic_ncols=True, leave=False
    ):
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
    cm = confusion_matrix(gt.flatten(), pred.flatten())
    # 创建图形和轴
    num_foreground_cls = len(class_idx_map)-1
    fig, ax = plt.subplots(figsize=(num_foreground_cls*2, num_foreground_cls))
    # 绘制混淆矩阵，每行独立进行颜色映射
    num_rows, num_cols = cm.shape
    for i in range(num_rows):
        row = cm[i].reshape(1, -1)
        im = ax.imshow(
            row,
            interpolation="nearest",
            cmap=CMAP_SEQ_COLOR[0],
            vmin=row.min(),
            vmax=row.max(),
            alpha=0.7,
            extent=(-0.5, num_cols-0.5, i-0.5, i+0.5),
        )
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    # 取消颜色条刻度
    cbar.set_ticks([])
    cbar.ax.set_ylabel("Column-Wise Color Mapping", rotation=270, labelpad=15)
    # 设置标题和标签
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    # 设置刻度和标签
    ticks = [
        {v: k for k, v in class_idx_map.items()}[idx]
        for idx in np.union1d(np.unique(gt), np.unique(pred))
    ]
    ax.set_xlim(-0.5, -0.5 + len(ticks))
    ax.set_xticks(np.arange(len(ticks)))
    ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.set_ylim(-0.5, -0.5 + len(ticks))
    ax.set_yticks(np.arange(len(ticks)))
    ax.set_yticklabels(ticks)
    # 添加数值标签
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="black",
            )
    # 调整布局并保存图像
    ax.set_aspect('auto')
    ax.set_box_aspect(0.5)  # 设置固定的纵横比
    fig.tight_layout()
    fig.savefig(save_path, dpi=500)
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
                values = sub_win[i][mask]
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
    index_class_map = {v: k for k, v in class_index_map.items()}
    num_subwins = sub_win.shape[0]
    unique_classes = sorted(np.unique(pred_array))[1:]
    num_rows = num_subwins  # 每个子图一行
    num_cols = 1  # 只有一列
    color = eval(f"plt.cm.{CMAP_SEQ_COLOR[0]}")(np.linspace(0, 1, len(unique_classes)))

    # 增加右侧空间以放置标题
    fig = plt.figure(figsize=(12, 1.2 * num_rows))

    # 创建子图列表
    axes = []
    for i in tqdm(
        range(num_subwins),
        desc="Drawing HU Violin Plot",
        dynamic_ncols=True,
        leave=False,
    ):
        ax = plt.subplot2grid((num_rows, num_cols), (i, 0))
        axes.append(ax)

        plot_data = []
        positions = []
        labels = []

        # 收集每个类别的数据
        for j, cls in enumerate(unique_classes):
            mask = pred_array == cls
            values = sub_win[i][mask]
            if len(values) > 0:
                plot_data.append(values)
                positions.append(j)
                labels.append(index_class_map[cls])

        # 绘制小提琴图
        try:
            parts = ax.violinplot(plot_data, positions=positions, widths=0.8)
        except Exception as e:
            print(f"Error in subwin {i}: {e}")
            return

        # 设置每个violin的颜色
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(color[j])
            pc.set_alpha(0.7)

        # 设置区间线的颜色
        for partname in ["cbars", "cmins", "cmaxes"]:
            if partname in parts:
                vp = parts[partname]
                vp.set_edgecolor(CMAP_COLOR[0])
                vp.set_linewidth(1)
                vp.set_alpha(0.3)

        # 在右侧添加垂直标题
        ax.text(
            1.02,
            0.5,
            f"A.Win. {i}",
            rotation=90,
            transform=ax.transAxes,
            verticalalignment="center",
        )

        ax.grid(True, axis="y")
        if i == num_subwins - 1:
            ax.set_xticks(range(len(unique_classes)))
            ax.set_xticklabels(
                [index_class_map[cls] for cls in unique_classes], rotation=90
            )
        else:
            ax.set_xticks([])
        ax.set_ylabel("Response")

    # 调整布局并保存，右侧留出更多空间放置标题
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
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
        pred_array = pred_array[..., ::-1, :]

    # visualization

    if wine_resps is not None:
        draw_combined_WinE_resp(
            image_array,
            wine_resps,
            os.path.join(save_root, "combined_wine", f"combined_wine_{exp_name}.png"),
        )
        draw_crsf_resp(
            all_window.cpu().numpy(),
            crsf_resp,
            num_windows,
            os.path.join(save_root, "crsf_resp", f"crsf_resp_{exp_name}.png"),
        )
        draw_HU_violinplot(
            image_array,
            gt_array,
            pred_array,
            wine_resps,
            class_idx_map,
            os.path.join(save_root, "HU_violin", f"HU_violin_{exp_name}.png"),
        )
    draw_confusion_matrix(
        gt_array,
        pred_array,
        class_idx_map,
        os.path.join(save_root, "cm", f"cm_{exp_name}.png"),
    )

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
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.6.1.NoNorm/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.6.1.NoNorm/MedNeXt/iter_200000.pth",
        class_idx_map=AbdomenCT1K_Map,
        itk_image_path="/file1/mgam_datasets/AbdomenCT_1K/spacing2_mha/image/01062.mha",
        itk_mask_path="/file1/mgam_datasets/AbdomenCT_1K/spacing2_mha/label/01062.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=False,
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
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.6.4.wl40_ww400/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.6.4.wl40_ww400/MedNeXt/iter_200000.pth",
        class_idx_map=AbdomenCT1K_Map,
        itk_image_path="/file1/mgam_datasets/AbdomenCT_1K/spacing2_mha/image/01062.mha",
        itk_mask_path="/file1/mgam_datasets/AbdomenCT_1K/spacing2_mha/label/01062.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=False,
        manual_win=True,
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
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.5.1.NoNorm/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.5.1.NoNorm/MedNeXt/iter_200000.pth",
        class_idx_map=KiTS23_Map,
        itk_image_path="/file1/mgam_datasets/KiTS23/spacing2_mha/image/00523.mha",
        itk_mask_path="/file1/mgam_datasets/KiTS23/spacing2_mha/label/00523.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=False,
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
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.5.4.wl20_ww400/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.5.4.wl20_ww400/MedNeXt/iter_200000.pth",
        class_idx_map=KiTS23_Map,
        itk_image_path="/file1/mgam_datasets/KiTS23/spacing2_mha/image/00523.mha",
        itk_mask_path="/file1/mgam_datasets/KiTS23/spacing2_mha/label/00523.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=False,
        manual_win=True,
    )

    # ImageTBAD
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.3.0.Window4_ImageTBAD/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.3.0.Window4_ImageTBAD/MedNeXt/best_Perf_mDice_iter_135000.pth",
        class_idx_map=ImageTBAD_Map,
        itk_image_path="/file1/mgam_datasets/ImageTBAD/spacing2_mha/image/180.mha",
        itk_mask_path="/file1/mgam_datasets/ImageTBAD/spacing2_mha/label/180.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=False,
        manual_win=False,
    )
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.3.1.Window8/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.3.1.Window8/MedNeXt/best_Perf_mDice_iter_130000.pth",
        class_idx_map=ImageTBAD_Map,
        itk_image_path="/file1/mgam_datasets/ImageTBAD/spacing2_mha/image/180.mha",
        itk_mask_path="/file1/mgam_datasets/ImageTBAD/spacing2_mha/label/180.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=False,
        manual_win=False,
    )
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.3.2.NoNorm/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.3.2.NoNorm/MedNeXt/iter_200000.pth",
        class_idx_map=ImageTBAD_Map,
        itk_image_path="/file1/mgam_datasets/ImageTBAD/spacing2_mha/image/180.mha",
        itk_mask_path="/file1/mgam_datasets/ImageTBAD/spacing2_mha/label/180.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=False,
        manual_win=False,
    )
    analyze_one_exp(
        config_path="/file1/mgam_projects/AutoWindow/configs/0.0.3.3.InstanceNorm/MedNeXt.py",
        ckpt_path="/file1/mgam_projects/AutoWindow/work_dirs/0.0.3.3.InstanceNorm/MedNeXt/iter_200000.pth",
        class_idx_map=ImageTBAD_Map,
        itk_image_path="/file1/mgam_datasets/ImageTBAD/spacing2_mha/image/180.mha",
        itk_mask_path="/file1/mgam_datasets/ImageTBAD/spacing2_mha/label/180.mha",
        save_root="/mnt/d/微云同步助手/312065559/微云同步/mgam_writing/AutoWindow/Figures",
        inst_norm=True,
        manual_win=False,
    )
