from mgamdata.dataset.Totalsegmentator.meta import CLASS_INDEX_MAP as Tsd_Map
from mgamdata.dataset.FLARE_2023.meta import CLASS_INDEX_MAP as FLARE23_Map
from analyzer import analyze_one_exp


if __name__ == "__main__":
    # Totalsegmentator
    Tsd_Interested_Classes = {
        k: Tsd_Map[k] 
        for k in ["background", "duodenum", "iliac_artery_left", "iliac_artery_right", "pancreas", "lung_upper_lobe_right"]
    }
    # analyze_one_exp(
    #     config_path="/home/zhangyq.sx/mgam_projects/AutoWindow/configs/1.0.3.2.InstanceNorm/MedNeXt.py", 
    #     ckpt_path="/fileser51/zhangyiqin.sx/mmseg/AutoWindow/work_dirs/1.0.3.2.InstanceNorm/MedNeXt/iter_500000.pth", 
    #     class_idx_map=Tsd_Interested_Classes, 
    #     itk_image_path="/home/zhangyq.sx/mgam_datasets/Totalsegmentator_Data/spacing2_mha/s1428/ct.mha",
    #     itk_mask_path="/home/zhangyq.sx/mgam_datasets/Totalsegmentator_Data/spacing2_mha/s1428/segmentations.mha",
    #     save_root="/home/zhangyq.sx/mgam_projects/AutoWindow/Figures",
    #     inst_norm=True,
    #     manual_win=False,
    #     invert_y=True,
    #     invert_channel=True,
    # )
    # analyze_one_exp(
    #     config_path="/home/zhangyq.sx/mgam_projects/AutoWindow/configs/1.0.3.1.Window4/MedNeXt.py", 
    #     ckpt_path="/fileser51/zhangyiqin.sx/mmseg/AutoWindow/work_dirs/1.0.3.1.Window4/MedNeXt/best_Perf_mDice_iter_285000.pth", 
    #     class_idx_map=Tsd_Interested_Classes, 
    #     itk_image_path="/home/zhangyq.sx/mgam_datasets/Totalsegmentator_Data/spacing2_mha/s1428/ct.mha",
    #     itk_mask_path="/home/zhangyq.sx/mgam_datasets/Totalsegmentator_Data/spacing2_mha/s1428/segmentations.mha",
    #     save_root="/home/zhangyq.sx/mgam_projects/AutoWindow/Figures",
    #     inst_norm=False,
    #     manual_win=False,
    #     invert_y=True,
    #     invert_channel=True,
    # )
    # analyze_one_exp(
    #     config_path="/home/zhangyq.sx/mgam_projects/AutoWindow/configs/1.0.3.0.Window8/MedNeXt.py", 
    #     ckpt_path="/fileser51/zhangyiqin.sx/mmseg/AutoWindow/work_dirs/1.0.3.0.Window8/MedNeXt/iter_480000.pth", 
    #     class_idx_map=Tsd_Interested_Classes, 
    #     itk_image_path="/home/zhangyq.sx/mgam_datasets/Totalsegmentator_Data/spacing2_mha/s1428/ct.mha",
    #     itk_mask_path="/home/zhangyq.sx/mgam_datasets/Totalsegmentator_Data/spacing2_mha/s1428/segmentations.mha",
    #     save_root="/home/zhangyq.sx/mgam_projects/AutoWindow/Figures",
    #     inst_norm=False,
    #     manual_win=False,
    #     invert_y=True,
    #     invert_channel=True,
    # )

    # FLARE_2023
    analyze_one_exp(
        config_path="/home/zhangyq.sx/mgam_projects/AutoWindow/configs/1.0.4.3.ww4096wl1024/MedNeXt.py", 
        ckpt_path="/fileser51/zhangyiqin.sx/mmseg/AutoWindow/work_dirs/1.0.4.3.ww4096wl1024/MedNeXt/best_Perf_mDice_iter_335000.pth", 
        class_idx_map=FLARE23_Map, 
        itk_image_path="/home/zhangyq.sx/mgam_datasets/FLARE_2023_Data/spacing2_mha/image/2200.mha",
        itk_mask_path="/home/zhangyq.sx/mgam_datasets/FLARE_2023_Data/spacing2_mha/label/2200.mha",
        save_root="/home/zhangyq.sx/mgam_projects/AutoWindow/Figures",
        inst_norm=False,
        manual_win=[1024,4096],
        invert_y=False,
        invert_channel=False,
    )
    analyze_one_exp(
        config_path="/home/zhangyq.sx/mgam_projects/AutoWindow/configs/1.0.4.1.Window4/MedNeXt.py", 
        ckpt_path="/fileser51/zhangyiqin.sx/mmseg/AutoWindow/work_dirs/1.0.4.1.Window4/MedNeXt/best_Perf_mDice_iter_445000.pth", 
        class_idx_map=FLARE23_Map, 
        itk_image_path="/home/zhangyq.sx/mgam_datasets/FLARE_2023_Data/spacing2_mha/image/2200.mha",
        itk_mask_path="/home/zhangyq.sx/mgam_datasets/FLARE_2023_Data/spacing2_mha/label/2200.mha",
        save_root="/home/zhangyq.sx/mgam_projects/AutoWindow/Figures",
        inst_norm=False,
        manual_win=None,
        invert_y=False,
        invert_channel=False,
    )
    analyze_one_exp(
        config_path="/home/zhangyq.sx/mgam_projects/AutoWindow/configs/1.0.4.0.Window8_FLARE2023/MedNeXt.py", 
        ckpt_path="/fileser51/zhangyiqin.sx/mmseg/AutoWindow/work_dirs/1.0.4.0.Window8_FLARE2023/MedNeXt/best_Perf_mDice_iter_215000.pth", 
        class_idx_map=FLARE23_Map, 
        itk_image_path="/home/zhangyq.sx/mgam_datasets/FLARE_2023_Data/spacing2_mha/image/2200.mha",
        itk_mask_path="/home/zhangyq.sx/mgam_datasets/FLARE_2023_Data/spacing2_mha/label/2200.mha",
        save_root="/home/zhangyq.sx/mgam_projects/AutoWindow/Figures",
        inst_norm=False,
        manual_win=None,
        invert_y=False,
        invert_channel=False,
    )