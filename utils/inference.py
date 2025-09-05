import os
import argparse
import traceback
import pdb
from colorama import Style, Fore

import torch
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

from itkit.mm.inference import Inferencer
from src.model import AutoWindowSetting



class Inferencer_3D(Inferencer):
    @torch.inference_mode()
    def Inference_FromNDArray(self, image_array:np.ndarray) -> torch.Tensor:
        data, _ = self._preprocess(image_array)
        model: AutoWindowSetting = self.model
        pred = model.predict(torch.stack(data['inputs']).cuda(), None)
        pred = torch.stack([b.seg_logits.data for b in pred]).cpu().squeeze(0)
        return pred # [Class, Z, Y, X]


def interactive_run(args):
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


def recursive_auto_inference(args):
    os.makedirs(args.output, exist_ok=True)
    inferencer = Inferencer_3D(args.config, args.checkpoint)
    print(Fore.BLUE, "Inferencer Initialized.", Style.RESET_ALL)
    for itk_image, itk_pred, mha_path in inferencer.Inference_FromITKFolder(args.input_root):
        output_path = os.path.join(args.output, os.path.basename(mha_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(itk_pred, output_path, useCompression=True)
        tqdm.write(f"Prediction saved to {output_path}.")


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--input-root', type=str, default=None)
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    # interactive_run(args)
    recursive_auto_inference(args)