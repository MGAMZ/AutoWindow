import os
import argparse
import traceback
import pdb

import torch
import SimpleITK as sitk
import numpy as np

from mgamdata.mm.inference import Inferencer
from mgamdata.mm.mmseg_Dev3D import EncoderDecoder_3D
from mgamdata.models.AutoWindow import AutoWindowSetting



def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--output', type=str, default=None)
    return parser.parse_args()


class Inferencer_3D(Inferencer):
    @torch.inference_mode()
    def Inference_FromNDArray(self, image_array:np.ndarray) -> torch.Tensor:
        inputs = torch.from_numpy(image_array).to(dtype=torch.float32, device='cuda')
        model: AutoWindowSetting = self.model
        image_meta = [{
            'ori_shape': image_array.shape,
            'img_shape': image_array.shape,
        }]
        pred = model.inference(inputs[None, None], image_meta).squeeze()
        return pred # [Class, Z, Y, X]


def main():
    args = parse_args()
    inferencer = Inferencer_3D(args.config, args.checkpoint)
    
    while True:
        user_input = input("Enter mha file to execute inference. Enter q to quit.")
        if user_input == "q":
            break
        if not os.path.exists(user_input):
            print(f"File does not exist: {user_input}")
            continue
        
        try:
            itk_image = sitk.ReadImage(user_input)
            print(f"Loaded mha file from {user_input}, size {itk_image.GetSize()[::-1]}.")
        except Exception as e:
            print(f"Error when sitk.ReadImage: {e}")
            print(traceback.print_exc())
            continue
        
        try:
            print("Inferencing start.")
            itk_image, itk_pred = inferencer.Inference_FromITK(itk_image)
            print("Inferencing finished.")
        except Exception as e:
            print(f"Error when inferencing: {e}")
            print(traceback.print_exc())
            continue
        
        if args.output is None:
            output_path = input(f"Inference finished for {user_input}. "
                                "\nEnter output path if required. "
                                "\nEnter n to skip saving.")
            if output_path == "n":
                continue
        else:
            output_path = os.path.join(args.output, os.path.basename(user_input))
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sitk.WriteImage(itk_pred, output_path)
            print(f"Prediction saved to {output_path}.")
        except Exception as e:
            print(f"Error saving file {user_input} prediction to {output_path}: {e}")
            continue



if __name__ == '__main__':
    main()