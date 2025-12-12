import numpy as np
import torch

import trimesh as tm
from plyfile import PlyData, PlyElement
from icecream import ic
import pandas as pd
import os
import sys
import cv2
import json
from argparse import ArgumentParser
import time
from natsort import natsorted
import glob
from MiniPLAS import sort_with_MiniPLAS
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import morton_sort


COORDS_SCALE = 255
RGB_SCALE = 255
C0 = 0.28209479177387814

def SH2RGB(sh):
    return sh*C0 + 0.5

def morton(a):
    x = a & 0x1FFFFF  
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x

def morton2DEncode(pos: torch.Tensor) -> torch.Tensor:
    x, y = pos.unbind(-1)
    answer = morton(x) | morton(y) << 1 
    return answer

def morton_sort_image(H, data):
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=data.device),
        torch.arange(H, device=data.device),
        indexing = 'ij'
    )
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    pos = torch.stack([x_flat, y_flat], dim = -1)
    morton_index = morton2DEncode(pos)
    sorted_indices = morton_index.argsort()
    xy_sorted = torch.stack([x_flat, y_flat], dim = 1)[sorted_indices]
    image = torch.zeros(H, H, data.shape[1], device=data.device, dtype=data.dtype)
    x, y = xy_sorted[:,0], xy_sorted[:, 1]
    image[y, x, :] = data
    return  image

def prune_gaussian(df, num_to_keep):
    scaling_act = np.exp
    opacity_act = lambda x: 1/(1+ np.exp(-x))
    df["impact"] = scaling_act((df["scale_0"] + df["scale_1"] + df["scale_2"]).astype(np.float64)) * opacity_act(df["opacity"].astype(np.float64))
    df = df.sort_values("impact", ascending = False)
    df = df.head(num_to_keep)
    return df

def gs_ply_to_df(ply_file, min_block_size):
    ply_load = tm.load(ply_file)
    ply_data = ply_load.metadata["_ply_raw"]["vertex"]["data"]
    df = pd.DataFrame(ply_data)
    num_gaussians = len(df)

    print("pruning for square PLAS") 
    sidelen = int(np.sqrt(num_gaussians))
    sidelen = sidelen // min_block_size * min_block_size
    df = prune_gaussian(df, sidelen * sidelen)

    return df, sidelen

def quantize_image(map, B, per_dim = 0):
    if per_dim == 0:
        x_min = map.min()
        x_max = map.max()
    else:
        x_min = map.min(axis = (0,1))
        x_max = map.max(axis = (0,1))
    eps = 1e-8
    scale = (2**B - 1) / (x_max - x_min + eps)
    map_q = np.clip(np.round((map - x_min) * scale), 0, 2**B - 1)
    return map_q, x_min, x_max

def RGB2SH(RGB):
    return (RGB - 0.5)/C0
    
def dequantize_image(map_q, B, x_min, x_max, per_dim = 0):
    if per_dim == 1:
        x_min = np.array(x_min)
        x_max = np.array(x_max)
    eps = 1e-8
    scale = (x_max - x_min + eps) / (2**B - 1)
    map_recon = map_q * scale + x_min
    return map_recon

def save_image(map, path, current_bd):

    if current_bd > 8:
        map = map.astype(np.uint16)
        mask = (1<<current_bd) - 1
        map = map & mask
        np_image = map.astype(np.uint16)
    else:
        np_image = map.astype(np.uint8)

    np_image = np_image[...,::-1]
    cv2.imwrite(path, np_image)
    
def pca_compression(data, k):
    M, N = data.shape
    if N<k:
        print(f"Target dimension {k}  should be smaller than original data {N}!")
    else:
        print(f"reduce dimension from {N} to {k}")
    
    mean = data.mean(axis = 0, keepdims = True)
    Xc = data - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k, :]
    Z = Xc @ components.T

    return Z, components, mean

def pca_recover(data, components, mean):
    re_data = data@components + mean
    ic(re_data.shape)
    return re_data

def split_xyz(xyz_q:np.ndarray, msb:int, overall_bd:int):
    assert 0<msb<overall_bd, f"MSB should between 0 and {overall_bd}"
    lsb = overall_bd - msb

    mask_lsb = (1<<lsb) - 1
    high = (xyz_q >> lsb).astype(np.uint16)
    low = (xyz_q & mask_lsb).astype(np.uint16)
    print(f"Split {overall_bd} bits into {msb} bits (high) and {lsb} bits (low)")
    return high, low, lsb

if __name__=="__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="PLAS generation script parameters")
    
    parser.add_argument("--gaussian_ply", type=str, default="frame000.ply")
    parser.add_argument("--plas_min_block_size", type=int, default=16)
    parser.add_argument("--result_path_save", type=str, default='./data/')
    parser.add_argument("--xyz_bd", type=int, default=20)
    parser.add_argument("--xyz_MSB", type=int, default=10)
    parser.add_argument("--f_dc_bd", type=int, default=10)
    parser.add_argument("--f_rest_bd", type=int, default=10)
    parser.add_argument("--opacity_bd", type=int, default=10)
    parser.add_argument("--scale_bd", type=int, default=10)
    parser.add_argument("--rot_bd", type=int, default=10)
    parser.add_argument("--radius_update_ratio", type=float, default=0.95) 
    parser.add_argument("--radius_decay", type=float, default=1) 
    parser.add_argument("--radius_iniSize_scale", type=float, default=0.95) 
    parser.add_argument("--plas_roll_fix", type=int, default=2)  # 0 for random shuffle, 1 for fix, 2 for no shuffle
    parser.add_argument("--PCAdim_for_AC", type = int, default=12)
    parser.add_argument("--mini_PLAS", type = int, default=1)
    parser.add_argument("--mini_PLAS_size", type = int, default=3) # maximum block size  = 2* this value + 2 = MBS in the paper
    parser.add_argument("--mini_PLAS_block_size", type = int, default=4) # minimal PLAS block size, 4*4 is default
    parser.add_argument("--mini_PLAS_single_scale", type = int, default=1) # 1: no progressive miniPLAS, only one scale
    parser.add_argument("--f_dc_clip", type = int, default=0) # 1: clip to 0-1 after convert dc_sh to RGB, 0 for no clipping, as well as larger bitstream
    args = parser.parse_args()




    print("3DGS to 2D map")

    gaussian_ply = args.gaussian_ply
    min_block_size = args.plas_min_block_size
    radius_update_ratio = args.radius_update_ratio
    radius_decay = args.radius_decay
    radius_iniSize_scale = args.radius_iniSize_scale
    plas_roll_fix = args.plas_roll_fix
    PCA_dim = args.PCAdim_for_AC
    mini_PLAS = args.mini_PLAS
    mini_PLAS_size = args.mini_PLAS_size
    mini_PLAS_block_size = args.mini_PLAS_block_size
    mini_PLAS_single_scale = args.mini_PLAS_single_scale
    f_dc_clip = args.f_dc_clip

    xyz_bd = args.xyz_bd
    xyz_MSB = args.xyz_MSB
    f_dc_bd = args.f_dc_bd
    f_rest_bd = args.f_rest_bd
    opacity_bd = args.opacity_bd
    scale_bd = args.scale_bd
    rot_bd = args.rot_bd
    target_bd = 16



    df, sidelen = gs_ply_to_df(gaussian_ply, min_block_size)
    print("resolution: ", sidelen)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"
    
    ic(device)



    image_path = args.result_path_save
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    args_dict = vars(args)
    config_file = os.path.join(image_path, "config.json")
    with open(config_file, "w") as f:
        json.dump(args_dict, f, indent=4)
    print(f"[INFO] save to: {config_file}")


    print("Dimensionality reduction for AC:")
    AC_data = df.loc[:, df.columns.str.startswith("f_rest")].values
    PCA_start_time = time.time()
    AC_cop, components, mean_ac = pca_compression(AC_data, PCA_dim)
    PCA_end_time = time.time()
    PCA_time = PCA_end_time - PCA_start_time
            
    new_cols = [f"f_rest_{i}" for i in range(PCA_dim)]
    df_AC = pd.DataFrame(AC_cop, columns=new_cols, index=df.index)
    df = df.drop(columns=df.columns[df.columns.str.startswith("f_rest")])
    df = pd.concat([df, df_AC], axis = 1)


    print("Two-stage Morton Scan mode")
    GS_data = torch.tensor(df.values)
    xyz = GS_data[:, 0:3]
    all_data = GS_data
    ic("3D Morton sort First.")
    morton_start_time = time.time()
    sorted_xyz, sorted_all = morton_sort(xyz, all_data, max_bits=16)
    morton_end_time = time.time()
    df.loc[:,:] = sorted_all.numpy()

    ori_data = torch.tensor(df.values, device = device)
    ic("2D Morton sort mapping.")
    mortonre_start_time = time.time()
    reverse_morton_image = morton_sort_image(sidelen, ori_data)
    mortonre_end_time = time.time()
    Morton2_time = mortonre_end_time - mortonre_start_time

    pre_data = reverse_morton_image.reshape(-1, df.values.shape[1])
    df.iloc[:,:] = pre_data.cpu().numpy()


             
    print("MiniPLAS module")

    coords_xyz = df[["x", "y", "z"]].values
    coords_xyz_min = coords_xyz.min()
    coords_xyz_range = coords_xyz.max() - coords_xyz_min
    coords_xyz_norm = (coords_xyz - coords_xyz_min) / coords_xyz_range
    coords_xyz_norm *= COORDS_SCALE
    coords_torch = torch.from_numpy(coords_xyz_norm).float().to(device)
        
    dc_vals = df.loc[:, df.columns.str.startswith("f_dc")].values
    rgb = np.clip(SH2RGB(dc_vals), 0, 1) * RGB_SCALE
    rgb_torch = torch.from_numpy(rgb).float().to(device)

    ac_vals = df.loc[:, df.columns.str.startswith("f_rest")].values
    ac_min = ac_vals.min()
    ac_max = ac_vals.max()
    ac_range = ac_max - ac_min
    ac_norm = (ac_vals - ac_min) / ac_range
    ac_norm*= RGB_SCALE
    ac_torch = torch.from_numpy(ac_norm).float().to(device)

    rot_vals = df.loc[:, df.columns.str.startswith("rot")].values
    rot_min = rot_vals.min()
    rot_max = rot_vals.max()
    rot_range = rot_max - rot_min
    rot_norm = (rot_vals - rot_min) / rot_range
    rot_norm*= COORDS_SCALE
    rot_torch = torch.from_numpy(rot_norm).float().to(device)

    scale_vals = df.loc[:, df.columns.str.startswith("scale")].values
    scale_min = scale_vals.min()
    scale_max = scale_vals.max()
    scale_range = scale_max - scale_min
    scale_norm = (scale_vals - scale_min) / scale_range
    scale_norm*= COORDS_SCALE
    scale_torch = torch.from_numpy(scale_norm).float().to(device)

    opt_vals = df.loc[:, df.columns.str.startswith("opacity")].values
    opt_min = opt_vals.min()
    opt_max = opt_vals.max()
    opt_range = opt_max - opt_min
    opt_norm = (opt_vals - opt_min) / opt_range
    opt_norm*= COORDS_SCALE
    opt_torch = torch.from_numpy(opt_norm).float().to(device)

    params = torch.cat([coords_torch, rgb_torch, ac_torch, opt_torch, scale_torch, rot_torch], dim = 1) # [N_gaussian, 6]

    params_torch_grid = params.permute(1, 0).reshape(-1, sidelen, sidelen)

    if mini_PLAS == 1:
        print(f"Mini_PLAS mode with max block size as {mini_PLAS_size*2+2}")
        plas_roll_fix = 2
        min_block_size = mini_PLAS_block_size
        radius_update_ratio = 0.5
    size_scale = 1

    PLAS_start_time = time.time()
    sorted_coords, sorted_grid_indices= sort_with_MiniPLAS(params_torch_grid, 
                                                        min_block_size, 
                                                        improvement_break=1e-4, 
                                                        verbose=True, 
                                                        size_scale= size_scale, 
                                                        radius_update_ratio = radius_update_ratio, 
                                                        radius_progressive = radius_decay, 
                                                        roll_fix = plas_roll_fix,
                                                        mini_PLAS=mini_PLAS,
                                                        mini_PLAS_size=mini_PLAS_size,
                                                        single_scale = mini_PLAS_single_scale)
    PLAS_end_time = time.time()

    sorted_indices = sorted_grid_indices.flatten().cpu().numpy()
    sorted_df = df.iloc[sorted_indices] #[N_Gaussian, 63]

    #save image
    xyz = sorted_df[["x", "y", "z"]].values.reshape(sidelen, sidelen, -1)  
    f_dc = sorted_df.loc[:, sorted_df.columns.str.startswith("f_dc")].values.reshape(sidelen, sidelen, -1)
    f_rest = sorted_df.loc[:, sorted_df.columns.str.startswith("f_rest")].values.reshape(sidelen, sidelen, -1)

    opacity = sorted_df[["opacity"]].values.reshape(sidelen, sidelen, -1)  
    scale = sorted_df.loc[:, sorted_df.columns.str.startswith("scale")].values.reshape(sidelen, sidelen, -1)
    rot = sorted_df.loc[:, sorted_df.columns.str.startswith("rot")].values.reshape(sidelen, sidelen, -1)

    preprocessing_start_time = time.time()

    xyz_q, xyz_min, xyz_max = quantize_image(xyz, xyz_bd)

    if f_dc_clip == 1:
        f_dc = np.clip(SH2RGB(f_dc), 0, 1)
        print("f_dc clip to 0-1 after SH2RGB, lower quality upper bound with smaller bitstream")
    else:
        f_dc = SH2RGB(f_dc)
        print("No f_dc clip after SH2RGB, higher quality upper bound with larger bitstream")
    f_dc_q, f_dc_min, f_dc_max = quantize_image(f_dc, f_dc_bd)
    f_rest_q, f_rest_min, f_rest_max = quantize_image(f_rest, f_rest_bd) #[M,M,45]
    opacity_q, opacity_min, opacity_max = quantize_image(opacity, opacity_bd)
    scale_q, scale_min, scale_max = quantize_image(scale, scale_bd)
    rot_q, rot_min, rot_max = quantize_image(rot, rot_bd)

    # split into 2
    name_xyz_1 = image_path + 'xyz_1.png'
    name_xyz_2 = image_path + 'xyz_2.png'
    if xyz_bd == 20:
        print("20 Bits for xyz")
        xyz_q = xyz_q.astype(np.uint32)
        high_xyz, low_xyz, xyz_LSB = split_xyz(xyz_q, xyz_MSB, xyz_bd)
    

    save_image(high_xyz, name_xyz_1, current_bd=xyz_MSB)
    save_image(low_xyz, name_xyz_2, current_bd=xyz_LSB)


    name_f_dc = image_path + 'f_dc.png'
    save_image(f_dc_q, name_f_dc, current_bd=f_dc_bd)



    AC_num = PCA_dim

    for i in range(AC_num//3):
        index = [i, i+ AC_num//3, i+ AC_num//3*2]
        f_rest_img = f_rest_q[:, :, index]
        name_f_rest = image_path + f'f_rest_{i:03d}.png'
        save_image(f_rest_img, name_f_rest, current_bd=f_rest_bd)

    name_opacity = image_path + 'opacity.png'

    # ic(opacity.shape) #[N, N, 1]
    opacity_padding = np.full((opacity_q.shape[0], opacity_q.shape[1], 1), (2**opacity_bd)//2, dtype = opacity_q.dtype)
    opacity_q = np.concatenate([ opacity_q,opacity_padding, opacity_padding], axis=2)


    save_image(opacity_q, name_opacity, current_bd=opacity_bd)

    name_scale = image_path + 'scale.png'
    save_image(scale_q, name_scale, current_bd=scale_bd)

    name_rotation = image_path + 'rotation_0.png'
    save_image(rot_q[:,:,0:3], name_rotation, current_bd=rot_bd)

    name_rotation = image_path + 'rotation_1.png'

    rot_padding = np.full((rot_q.shape[0], rot_q.shape[1], 1), (2**rot_bd)//2, dtype = rot_q.dtype)
    rot_q_p = np.concatenate([ rot_q[:,:,3:4],rot_padding, rot_padding], axis=2)

    save_image(rot_q_p, name_rotation, current_bd=rot_bd)

    preprocessing_end_time = time.time()

    print(f"Saving image to {image_path}")

    meta_data = image_path + 'metadata.json'
    

    data = {
        "xyz_min": xyz_min,
        "xyz_max": xyz_max,
        "xyz_bd": xyz_bd,
        "xyz_MSB": xyz_MSB,
        "f_dc_min": f_dc_min,
        "f_dc_max": f_dc_max,
        "f_dc_bd": f_dc_bd,
        "f_rest_min": f_rest_min,
        "f_rest_max": f_rest_max,
        "f_rest_bd": f_rest_bd,
        "opacity_min": opacity_min,
        "opacity_max":  opacity_max,
        "opacity_bd": opacity_bd,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "scale_bd": scale_bd,
        "rot_min": rot_min,
        "rot_max": rot_max,
        "rot_bd": rot_bd,
        "AC_dim":PCA_dim,
    }
    
    data = {k: float(v) if isinstance(v, np.floating) else v for k, v in data.items()}

    with open(meta_data, "w") as f:
        json.dump(data, f, indent=4)

    
    time_data_file = image_path + 'time.json'
    
    time_data = {}
    time_data["3D Morton_time: "] = morton_end_time - morton_start_time
    time_data["2D Morton_time: "] = Morton2_time
    time_data["PLAS_time: "] = PLAS_end_time - PLAS_start_time
    time_data["PCA time "] = PCA_time
    
    time_data["Preprocessing_time (quantization, save image): "] = preprocessing_end_time - preprocessing_start_time
    
    with open(time_data_file, "w") as f:
        json.dump(time_data, f, indent=4)
    

    PCA_components_file = image_path + 'pca_AC_all.json'
    print("Saving PCA components for PCA AC(all)")
    if not isinstance(components, list):
        components = components.tolist()
    if not isinstance(mean_ac, list):
        mean_ac = mean_ac.tolist()
    PCA_AC_all_data = {
        'components': components,
        'mean': mean_ac,
    }
    with open(PCA_components_file, "w") as f:
        json.dump(PCA_AC_all_data, f, indent=4)
    



        
        





   
    
    
    

