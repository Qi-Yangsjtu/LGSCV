#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
from scene import Scene

from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision

from argparse import ArgumentParser
from arguments import PipelineParams
from gaussian_renderer import GaussianModel
from utils.camera_utils import JSON_to_camera
from icecream import ic


def render_sets(pipeline : PipelineParams, config, model_path):
    
    cameras = JSON_to_camera(config)
    cameras.sort(key = lambda x: x.image_name)

    with torch.no_grad():
        gaussians = GaussianModel()
        gaussians.load_ply(model_path)
        
        bg_color = [1,1,1] if args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_path = config.result_root
        makedirs(render_path, exist_ok=True)
        for idx, view in enumerate(tqdm(cameras, desc="Rendering progress")):
            if args.test_camera == -1:
                rendering = render(view, gaussians, pipeline, background)["render"]
                torchvision.utils.save_image(rendering, os.path.join(render_path, "{:05d}{}.png".format(idx+1, config.png_suffix)))
            else:
                if idx+1 in args.test_camera:
                    rendering = render(view, gaussians, pipeline, background)["render"]
                    torchvision.utils.save_image(rendering, os.path.join(render_path, "{:05d}{}.png".format(idx+1, config.png_suffix)))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
   
    parser.add_argument("--camera_json", type=str, default="cameras.json")
    
    parser.add_argument("--model", type=str, default="frame000.ply")
    parser.add_argument("--result_root", type=str, default="./data/")

    parser.add_argument("--png_suffix", type=str, default="")
    parser.add_argument("--test_camera", nargs="+", type=int, default=[1, 9, 10]) #bartender and cinema is 9/11, breakfast is 6/8
    args = parser.parse_args()
    
    args.resolution = 1
    args.sh_degree=3
    args.white_background=False
    args.data_device='cuda'

     
    
    render_sets(pipeline.extract(args), args, args.model)
