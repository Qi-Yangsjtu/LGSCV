import numpy
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import torch
from argparse import ArgumentParser
import glob
from natsort import natsorted
from icecream import ic
import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim
from PIL import Image
import json
import torchvision.transforms.functional as tf
from tqdm import tqdm
from pathlib import Path

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    png_file = [f for f in os.listdir(renders_dir) if f.lower().endswith('.png')]
    ic(png_file)
    for fname in png_file:
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--ref_img", default='./data/', type=str)
    parser.add_argument("--dis_img", default='./data/', type=str)
    parser.add_argument("--result_path", default='./data/', type=str)

    args = parser.parse_args()

    ref_img_path = Path(args.ref_img)
    dis_img_path = Path(args.dis_img)
    renders, gts, image_names = readImages(dis_img_path, ref_img_path)

    ssims = []
    psnrs = []
    lpipss = []

    loss_LPIPS = lpips.LPIPS(net = 'vgg').cuda()
    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(loss_LPIPS(renders[idx], gts[idx]))

    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("")

    full_dict = {}
    per_view_dict = {}

    full_dict[args.dis_img] = ({"SSIM": torch.tensor(ssims).mean().item(),
                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                            "LPIPS": torch.tensor(lpipss).mean().item()})
    per_view_dict[args.dis_img]=({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    file1 = os.path.join(args.result_path, "Overall_Metric.json")
    file2 = os.path.join(args.result_path, "Per_View_Metric.json")

    with open(file1, 'w') as fp:
        json.dump(full_dict, fp, indent=True)
    with open(file2, 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)


    

