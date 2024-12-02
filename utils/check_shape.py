import os
import multiprocessing as mp
import pdb
from tqdm import tqdm

import SimpleITK as sitk


def check_a_sample(args):
    itk_image_path, itk_label_path = args
    itk_image = sitk.ReadImage(itk_image_path)
    itk_label = sitk.ReadImage(itk_label_path)
    image_shape = itk_image.GetSize()
    label_shape = itk_label.GetSize()
    if image_shape != label_shape:
        return (itk_image_path, itk_label_path, image_shape, label_shape)
    else:
        return None


def main():
    data_root = '/home/zhangyq.sx/FLARE_2023_Data/spacing2_mha'
    image_root = os.path.join(data_root, 'image')
    label_root = os.path.join(data_root, 'label')
    image_list = os.listdir(image_root)
    task_list = []
    
    for image_name in image_list:
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(label_root, image_name)
        if not os.path.exists(label_path):
            print(f'Label not found: {label_path}')
            continue
        task_list.append((image_path, label_path))
        
    with mp.Pool(8) as pool:
        results = pool.imap_unordered(check_a_sample, task_list)
        for result in tqdm(results, total=len(task_list)):
            if result is not None:
                tqdm.write(str(result))


if __name__ == "__main__":
    main()