import numpy as np
import torch
from plyfile import PlyData, PlyElement
from icecream import ic
import os
import sys
import cv2
import glob
from natsort import natsorted
import json
from argparse import ArgumentParser
from pathlib import Path
import time
from GSTo2DMap import pca_recover
from pathlib import Path
import lzma
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class GS:
    def __init__(self, center=None, normal=None, baseColor=None, R_SH=None, G_SH=None, B_SH=None, opacity=None, scale=None, rotate=None):
        self.center = center
        self.normal = normal
        self.baseColor = baseColor
        self.R_SH = R_SH
        self.G_SH = G_SH
        self.B_SH = B_SH
        self.opacity = opacity
        self.scale = scale
        self.rotate = rotate

    def construct_list_of_attributes(self):
        data = np.hstack((self.R_SH, self.G_SH, self.B_SH))
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.baseColor.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(data.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotate.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def write_3DG_to_PLY_binary(self, filedir, flag_binary):
        # 1 for binary, 0 for ASCII
        if os.path.exists(filedir):
            os.system('rm '+ filedir)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        xyz = self.center

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        data = np.hstack((self.center, self.normal, self.baseColor, self.R_SH, self.G_SH, self.B_SH))
        data = np.hstack((data, np.expand_dims(self.opacity, axis=1)))
        attributes = np.hstack((data, self.scale, self.rotate))

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        if flag_binary == 1:
            PlyData([el]).write(filedir)
        else:
            PlyData([el], text = True).write(filedir)

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

C0 = 0.28209479177387814

def RGB2SH(RGB):
    return (RGB - 0.5)/C0
    
def SH2RGB(sh):
    return sh*C0 + 0.5

def dequantize_image(map_q, B, x_min, x_max, per_dim = 0):
    if per_dim == 1:
        x_min = np.array(x_min)
        x_max = np.array(x_max)
    eps = 1e-8
    scale = (x_max - x_min + eps) / (2**B - 1)
    map_recon = map_q * scale + x_min
    return map_recon

def RGB2YUV(img, bits, yuv_img):
    rgb16 = cv2.imread(img, cv2.IMREAD_UNCHANGED) #[B, G, R] becasue we using imread
    rgb16 = rgb16[...,::-1]


    y = rgb16[:,:,0]
    u = rgb16[:,:,1]
    v = rgb16[:,:,2]


    with open(yuv_img, 'wb') as f_yuv:
        if bits>8:
            f_yuv.write((y.astype(np.uint16) ).astype('<u2').tobytes())
            f_yuv.write((u.astype(np.uint16) ).astype('<u2').tobytes())
            f_yuv.write((v.astype(np.uint16) ).astype('<u2').tobytes())
        else:
            f_yuv.write((y.astype(np.uint8) ).tobytes())
            f_yuv.write((u.astype(np.uint8) ).tobytes())
            f_yuv.write((v.astype(np.uint8) ).tobytes())

def merge_xyz(high:np.ndarray, low:np.ndarray, lsb_bits:int):
    return ((high.astype(np.uint32)<<lsb_bits)| low.astype(np.uint32)).astype(np.uint32)

def rec_gs(dec_image_file, height, width, meta_data, reconstruction_path, PCA_meta):

    gs_data = torch.zeros((height, width, 62), dtype=torch.float32)
    postprocess_start_time = time.time()
    index = 0

    PCA_dim = meta_data["AC_dim"]

    de_f_dc = cv2.imread(dec_image_file[index], cv2.IMREAD_UNCHANGED)
    de_f_dc = de_f_dc[...,::-1]
    max_value = 2**f_dc_original_bd - 1
    de_f_dc = np.clip(de_f_dc, 0, max_value)
    de_f_dc = de_f_dc.astype(np.float32)
    de_f_dc = dequantize_image(de_f_dc, meta_data["f_dc_bd"], meta_data["f_dc_min"], meta_data["f_dc_max"], per_dim = 1)
    de_f_dc = RGB2SH(de_f_dc)
    gs_data[:,:,6:9] = torch.from_numpy(de_f_dc)

    index_st = 1
    index_end = PCA_dim//3
    tensor_f_rest = torch.zeros((height, width, 3, PCA_dim//3), dtype=torch.float32)
    for i in range(index_st, index_end+1):
        img = cv2.imread(dec_image_file[i], cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32)
        img = img[...,::-1].copy()
        tensor_f_rest[:,:,:,i-1] = torch.from_numpy(img)

    tensor_f_rest = tensor_f_rest.permute(2,3,0,1).reshape(PCA_dim, height, width).permute(1,2,0)
    max_value = 2**f_rest_original_bd - 1
    tensor_f_rest = torch.clip(tensor_f_rest, 0, max_value)
    tensor_f_rest = dequantize_image(tensor_f_rest, meta_data["f_rest_bd"], meta_data["f_rest_min"], meta_data["f_rest_max"], per_dim = 1)
    
    components = torch.tensor(PCA_meta["components"], dtype=torch.float32)
    mean_AC = torch.tensor(PCA_meta["mean"], dtype=torch.float32)
    AC_re = pca_recover(tensor_f_rest, components, mean_AC)
    tensor_f_rest = AC_re
    
    gs_data[:,:,9:54] = tensor_f_rest

    index = 1 + PCA_dim//3
    de_opacity = cv2.imread(dec_image_file[index], cv2.IMREAD_UNCHANGED)
    de_opacity = de_opacity[...,::-1]
    de_opacity = de_opacity[:,:,0]

    max_value = 2**opacity_original_bd - 1
    de_opacity = np.clip(de_opacity, 0, max_value)
    de_opacity = de_opacity.astype(np.float32)
    de_opacity = dequantize_image(de_opacity, meta_data["opacity_bd"], meta_data["opacity_min"], meta_data["opacity_max"], per_dim = 1)
    gs_data[:,:,54:55] = torch.from_numpy(de_opacity).unsqueeze(2)

    index = 2+ PCA_dim//3
    de_rotation = cv2.imread(dec_image_file[index], cv2.IMREAD_UNCHANGED)
    de_rotation = de_rotation[...,::-1]
    max_value = 2**f_dc_original_bd - 1
    de_rotation = np.clip(de_rotation, 0, max_value)
    de_rotation = de_rotation.astype(np.float32)
    de_rotation = dequantize_image(de_rotation, meta_data["rot_bd"], meta_data["rot_min"], meta_data["rot_max"])
    gs_data[:,:,58:61] = torch.from_numpy(de_rotation)


    de_rotation = cv2.imread(dec_image_file[index+1], cv2.IMREAD_UNCHANGED)
    de_rotation = de_rotation[...,::-1]
    de_rotation = de_rotation[:,:,0]
    de_rotation = np.clip(de_rotation, 0, max_value)
    de_rotation = de_rotation.astype(np.float32)
    de_rotation = dequantize_image(de_rotation, meta_data["rot_bd"], meta_data["rot_min"], meta_data["rot_max"])
    gs_data[:,:,61:62] = torch.from_numpy(de_rotation).unsqueeze(2)

    index = 4+PCA_dim//3
    de_scale = cv2.imread(dec_image_file[index], cv2.IMREAD_UNCHANGED)
    de_scale = de_scale[...,::-1]
    max_value = 2**scale_original_bd - 1
    de_scale = np.clip(de_scale, 0, max_value)
    de_scale = de_scale.astype(np.float32)
    de_scale = dequantize_image(de_scale, meta_data["scale_bd"], meta_data["scale_min"], meta_data["scale_max"], per_dim = 1)
    gs_data[:,:,55:58] = torch.from_numpy(de_scale)

    index = 5+PCA_dim//3
    xyz_original_bd = meta_data["xyz_bd"]
    xyz_MSB = meta_data["xyz_MSB"]
    xyz_LSB = xyz_original_bd - xyz_MSB

    de_xyz_1 = cv2.imread(dec_image_file[index], cv2.IMREAD_UNCHANGED)
    de_xyz_1 = de_xyz_1[...,::-1]
    max_value = 2**xyz_MSB - 1
    de_xyz_1 = np.clip(de_xyz_1, 0, max_value)

    de_xyz_2 = cv2.imread(dec_image_file[index+1], cv2.IMREAD_UNCHANGED)
    de_xyz_2 = de_xyz_2[...,::-1]
    max_value = 2**xyz_LSB - 1
    de_xyz_2 = np.clip(de_xyz_2, 0, max_value)

    de_xyz = merge_xyz(de_xyz_1, de_xyz_2, xyz_LSB)
    de_xyz = de_xyz.astype(np.float32)


    de_xyz = dequantize_image(de_xyz, meta_data["xyz_bd"], meta_data["xyz_min"], meta_data["xyz_max"], per_dim = 1)
    
    gs_data[:,:,0:3] = torch.from_numpy(de_xyz)

    gs_data = gs_data.reshape(-1, 62)

    gs_object = GS(center=gs_data[:,0:3],
                normal=gs_data[:,3:6],
                baseColor=gs_data[:,6:9],
                R_SH=gs_data[:,9:24],
                G_SH=gs_data[:,24:39],
                B_SH=gs_data[:,39:54],
                opacity=gs_data[:,54],
                scale=gs_data[:, 55:58],
                rotate=gs_data[:, 58:62])
    postprocess_end_time = time.time()
    gs_file = reconstruction_path + '/point_cloud.ply'
    gs_object.write_3DG_to_PLY_binary(gs_file, flag_binary=1)
    return postprocess_end_time - postprocess_start_time

def generate_YUV(file, name, path,PCA_dim):
    print("Generate YUV for GS sequence")
    

    data_path = file + name
    image_file = natsorted(glob.glob(f"{data_path}/*.png"))

    with open(data_path + '/metadata.json', "r") as f:
        meta_data = json.load(f)
        

    f_dc_original_bd = meta_data["f_dc_bd"]
    f_rest_original_bd = meta_data["f_rest_bd"]
    opacity_original_bd = meta_data["opacity_bd"]
    rotation_original_bd = meta_data["rot_bd"]
    scale_original_bd = meta_data["scale_bd"]
    xyz_original_bd = meta_data["xyz_bd"]

    index = 0 # for f_dc
    f_dc_yuv_file = path + 'f_dc_yuv.yuv'

    RGB2YUV(image_file[index], f_dc_original_bd,  f_dc_yuv_file)

    index_st = 1
    index_end = PCA_dim//3
    

    for j in range(index_st, index_end+1):
        f_rest_yuv_file = path + f'f_rest_{j}_yuv.yuv'
        RGB2YUV(image_file[j], f_rest_original_bd,  f_rest_yuv_file)

    index = 1 + PCA_dim//3 # for opacity
    opacity_yuv_file = path + 'opacity_yuv.yuv'

    RGB2YUV(image_file[index], opacity_original_bd,  opacity_yuv_file)

    # index = 17
    index = 2 + PCA_dim//3
    rotation_yuv_file_1 = path + 'rotation_1_yuv.yuv'

    RGB2YUV(image_file[index], rotation_original_bd,  rotation_yuv_file_1)

    rotation_yuv_file_2 = path + 'rotation_2_yuv.yuv'
    RGB2YUV(image_file[index+1], rotation_original_bd,  rotation_yuv_file_2)

        # index = 19
    index = 4 + PCA_dim//3
    scale_yuv_file = path + 'scale_yuv.yuv'
    RGB2YUV(image_file[index], scale_original_bd,  scale_yuv_file)

        # index = 20
    index = 5 + PCA_dim//3
    xyz_1_yuv_file = path + 'xyz_1_yuv.yuv'
    xyz_MSB = meta_data["xyz_MSB"]
    xyz_LSB = xyz_original_bd - xyz_MSB

    RGB2YUV(image_file[index], xyz_MSB,  xyz_1_yuv_file)
    
    xyz_2_yuv_file = path + 'xyz_2_yuv.yuv'
    RGB2YUV(image_file[index+1], xyz_LSB,  xyz_2_yuv_file)

def HEVC_lossy(input_yuv, 
               frames, 
               height, 
               width, 
               bitrate_path, 
               original_bd, 
               qp, 
               yuv_mode, 
               HM_encoder, 
               HM_decoder, 
               HM_cfg,
               max_CU = 64,
               CU_depth = 4):


    outbin_path = bitrate_path+input_yuv.split("/")[-1].replace("_yuv.yuv", ".bin")
    decodedyuv_path = outbin_path.replace(".bin", "_de.yuv")


    cfg = 'encoder_intra_main_rext.cfg'
    if max_CU == 64:
        QTL = 5
    else:
        QTL = int(np.log2(max_CU))

    cmd_hevc_encode = (
    f"{HM_encoder} "
    f"-c {HM_cfg}{cfg} "
    f"--InputFile={input_yuv} "
    f"--BitstreamFile={outbin_path} "
    f"--SourceWidth={width} "
    f"--SourceHeight={height} "
    f"--InputBitDepth={original_bd} "
    f"--InternalBitDepth={original_bd} "
    f"--OutputBitDepth={original_bd} "
    f"--InputChromaFormat={yuv_mode} "
    f"--FrameRate=30 "
    f"--FramesToBeEncoded={frames} "
    f"--QP={qp} "
    f"--MaxCUWidth={max_CU} "
    f"--MaxCUHeight={max_CU} "
    f"--MaxPartitionDepth={CU_depth} "
    f"--QuadtreeTULog2MaxSize={QTL} "
    f"--ConformanceWindowMode=1 "
    ) 
    print("\n--------hevc coding-----------")
    encode_start_time = time.time()
    os.system(cmd_hevc_encode)
    encode_end_time = time.time()

    cmd_hevc_decode = (
    f"{HM_decoder} "
    f"--BitstreamFile={outbin_path} "
    f"--ReconFile={decodedyuv_path} "
    )
    ic(cmd_hevc_encode)
    print("\n--------hevc decoding-----------")
    decode_start_time = time.time()

    os.system(cmd_hevc_decode)

    decode_end_time = time.time()

    f_size = Path(outbin_path).stat().st_size
    bpp = f_size / (height * width)
    return bpp, encode_end_time - encode_start_time, decode_end_time - decode_start_time

def HEVC_lossless(input_yuv, 
                  frames, 
                  height, 
                  width, 
                  bitrate_path, 
                  original_bd, 
                  yuv_mode, 
                  HM_encoder, 
                  HM_decoder, 
                  HM_cfg,
                  max_CU = 64,
                  CU_depth = 4):

    outbin_path = bitrate_path+input_yuv.split("/")[-1].replace("_yuv.yuv", ".bin")
    decodedyuv_path = outbin_path.replace(".bin", "_de.yuv")

    cfg = HM_cfg + "encoder_intra_main_rext_lossless.cfg"

    if max_CU == 64:
        QTL = 5
    else:
        QTL = int(np.log2(max_CU))
    
    cmd_hevc_encode = (
    f"{HM_encoder} "
    f"-c {cfg} "
    f"--InputFile={input_yuv} "
    f"--BitstreamFile={outbin_path} "
    f"--SourceWidth={width} "
    f"--SourceHeight={height} "
    f"--InputBitDepth={original_bd} "
    f"--InternalBitDepth={original_bd} "
    f"--OutputBitDepth={original_bd} "
    f"--InputChromaFormat={yuv_mode} "
    f"--FrameRate=30 "
    f"--FramesToBeEncoded={frames} "
    f"--QP=0 "
    f"--CostMode=lossless "
    f"--ExtendedPrecision=1"
    f"--TransquantBypassEnableFlag=1 "
    f"--CUTransquantBypassFlagForce=1 "
    f"--IntraReferenceSmoothing=0 "
    f"--MaxCUWidth={max_CU} "
    f"--MaxCUHeight={max_CU} "
    f"--MaxPartitionDepth={CU_depth} "
    f"--QuadtreeTULog2MaxSize={QTL} "
    f"--ConformanceWindowMode=1 "
    )   
    ic(cmd_hevc_encode)
    ic(cfg)
    print("\n--------hevc coding-----------")

    encode_start_time = time.time()
    os.system(cmd_hevc_encode)
    encode_end_time = time.time()

    print("\n--------hevc decoding-----------")

    cmd_hevc_decode = (
    f"{HM_decoder} "
    f"--BitstreamFile={outbin_path} "
    f"--ReconFile={decodedyuv_path} "
    )
    decode_start_time = time.time()
    os.system(cmd_hevc_decode)
    decode_end_time = time.time()

    f_size = Path(outbin_path).stat().st_size
    bpp = f_size / (height * width)

    return bpp, encode_end_time - encode_start_time, decode_end_time - decode_start_time

def rec_image(input_YUV, img_path, bits, width, height):
    ic(input_YUV)

    if bits<=8:
        dtype_read = np.uint8
        bytes_per_sample = 1
    else:
        dtype_read = np.dtype('<u2')
        bytes_per_sample = 2

    data_y = []
    data_u = []
    data_v = []
    with open(input_YUV, 'rb') as f_rec:
        frame_size_bytes = width * height * bytes_per_sample

        y_plane = np.frombuffer(f_rec.read(frame_size_bytes), dtype=dtype_read).reshape(height, width)
        u_plane = np.frombuffer(f_rec.read(frame_size_bytes), dtype=dtype_read).reshape(height, width)
        v_plane = np.frombuffer(f_rec.read(frame_size_bytes), dtype=dtype_read).reshape(height, width)
        data_y.append(y_plane)
        data_u.append(u_plane)
        data_v.append(v_plane)

    
    y = (data_y[0]).astype(np.uint16 if bits > 8 else np.uint8)
    u = (data_u[0]).astype(np.uint16 if bits > 8 else np.uint8)
    v = (data_v[0]).astype(np.uint16 if bits > 8 else np.uint8)

    rgb16_out = np.stack([y, u, v], axis = -1)
        
        
    rgb16_out = rgb16_out[..., ::-1]
    
    img_save_path = img_path + f"{000:03d}" + '/' + input_YUV.split("/")[-1].replace("_de.yuv", ".png")
    ic(img_save_path)
    cv2.imwrite(img_save_path, rgb16_out)

if __name__=="__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="3DGS 2D map HEVC compression script parameters")
    parser.add_argument("--sequence_path", type=str, default='./data/')
    parser.add_argument("--bitrate_path", type=str, default='./data/')
    parser.add_argument("--HM_encoder_binary", type=str, default='./HM/TAppEncoderStatic')
    parser.add_argument("--HM_decoder_binary", type=str, default='./HM/TAppDecoderStatic')
    parser.add_argument("--HM_cfg_path", type=str, default='./HM/cfg/')
    parser.add_argument("--reconstruction_path", type=str, default='./data/')
    parser.add_argument("--reconstruction_img_path", type=str, default='./data/')
    parser.add_argument("--f_rest_1_yuv_mode", type=int, default=0)
    parser.add_argument("--xyz_qp", type=int, default=0)
    parser.add_argument("--f_dc_qp", type=int, default=0)
    parser.add_argument("--f_rest_1_qp", type=int, default=0)
    parser.add_argument("--opacity_qp", type=int, default=0)
    parser.add_argument("--scale_qp", type=int, default=0)
    parser.add_argument("--rot_qp", type=int, default=0)
    parser.add_argument("--max_CU", type=int, default=64)
    parser.add_argument("--CU_depth", type=int, default=4)


    args = parser.parse_args()

    print("Compress PLAS image via 2D codec.")
    sequence_path = args.sequence_path
    frames = natsorted([f for f in os.listdir(sequence_path) if os.path.isdir(os.path.join(sequence_path, f))])
    frames = frames[0]
    num_frames = 1


    bitrate_path = args.bitrate_path
    HM_encoder = args.HM_encoder_binary
    HM_decoder = args.HM_decoder_binary
    HM_cfg = args.HM_cfg_path
    max_CU = args.max_CU
    CU_depth = args.CU_depth


    f_rest_1_yuv_mode = int(args.f_rest_1_yuv_mode)


    save_path = bitrate_path 
    create_folder(save_path)
    reconstruction_path = args.reconstruction_path
    create_folder(reconstruction_path)
    img_reconstruction_path = args.reconstruction_img_path
    create_folder(img_reconstruction_path)

    args_dict = vars(args)
    config_file = os.path.join(save_path, "config.json")
    with open(config_file, "w") as f:
        json.dump(args_dict, f, indent=4)
    print(f"[INFO] save to: {config_file}")
    
    for i in range(num_frames):
        folder_name = f"{i:03d}"
        folder_path = os.path.join(img_reconstruction_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        folder_path = os.path.join(reconstruction_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)




    img = cv2.imread(sequence_path+frames+'/f_dc.png', cv2.IMREAD_UNCHANGED)
    height, width = img.shape[:2] #[H, W, (B, G, R)]

    with open(sequence_path+frames + '/metadata.json', "r") as f:
        meta_data = json.load(f)
    


    f_dc_original_bd = meta_data["f_dc_bd"]
    f_rest_original_bd = meta_data["f_rest_bd"]
    opacity_original_bd = meta_data["opacity_bd"]
    rotation_original_bd = meta_data["rot_bd"]
    scale_original_bd = meta_data["scale_bd"]
    xyz_original_bd = meta_data["xyz_bd"]
    xyz_MSB = meta_data["xyz_MSB"]
    xyz_LSB = xyz_original_bd - xyz_MSB
    PCA_dim = meta_data["AC_dim"]



    flag_generate_yuv = True
    if flag_generate_yuv:
        generate_YUV(sequence_path, frames, bitrate_path, PCA_dim)


    flag_xyz = True
    flag_f_dc = True
    flag_f_rest = True
    flag_opacity = True
    flag_scaling = True
    flag_rotation = True

    # compress f_dc image
    if flag_f_dc:
        print("\n--------hevc coding (f_dc)-----------")
        f_dc_yuv_file = bitrate_path + 'f_dc_yuv.yuv'
        f_dc_qp = args.f_dc_qp
        f_dc_yuv_mode = '444'
        f_dc_bpp, f_dc_enc_time, f_dc_dec_time = HEVC_lossy(f_dc_yuv_file, num_frames, height, width, bitrate_path, f_dc_original_bd, f_dc_qp, f_dc_yuv_mode, HM_encoder, HM_decoder, HM_cfg,max_CU, CU_depth)
        f_dc_yuv_lossy = 0
        rec_image(f_dc_yuv_file.replace("_yuv.yuv", "_de.yuv"), img_reconstruction_path, f_dc_original_bd, width, height)
    # compress f_rest image
    if flag_f_rest:
        index_st = 1
        index_end = PCA_dim//3
        f_rest_1_qp = args.f_rest_1_qp
        f_rest_bpp = []
        f_rest_enc_time = []
        f_rest_dec_time = []
        print("\n--------hevc coding (f_rest)-----------")
        for i in range(index_st, index_end+1):
            f_rest_yuv_file = bitrate_path + f'f_rest_{i}_yuv.yuv'
            f_rest_qp = f_rest_1_qp
            f_rest_yuv_lossy = f_rest_1_yuv_mode
            f_rest_yuv_mode = '444'
            sub_f_rest_bpp, sub_f_rest_enc_time, sub_f_rest_dec_time = HEVC_lossy(f_rest_yuv_file, num_frames, height, width, bitrate_path, f_rest_original_bd, f_rest_qp, f_rest_yuv_mode, HM_encoder, HM_decoder, HM_cfg, max_CU, CU_depth)
            f_rest_bpp.append(sub_f_rest_bpp)
            f_rest_enc_time.append(sub_f_rest_enc_time)
            f_rest_dec_time.append(sub_f_rest_dec_time)
            rec_image(f_rest_yuv_file.replace("_yuv.yuv", "_de.yuv"), img_reconstruction_path, f_rest_original_bd, width, height)
        f_rest_file_size = sum(f_rest_bpp)
        f_rest_enc_time_all = sum(f_rest_enc_time)
        f_rest_dec_time_all = sum(f_rest_dec_time)

    # compress opacity image
    if flag_opacity:
        opacity_yuv_file = bitrate_path + 'opacity_yuv.yuv'
        opacity_qp = args.opacity_qp
        opacity_yuv_mode = '444'
        opacity_yuv_lossy = 0
        print("\n--------hevc coding (opacity)-----------")
        opacity_bpp, opacity_enc_time, opacity_dec_time = HEVC_lossy(opacity_yuv_file, num_frames, height, width, bitrate_path, opacity_original_bd, opacity_qp, opacity_yuv_mode, HM_encoder, HM_decoder, HM_cfg, max_CU, CU_depth)
        rec_image(opacity_yuv_file.replace("_yuv.yuv", "_de.yuv"), img_reconstruction_path, opacity_original_bd, width, height)
    
    # compress rotation
    if flag_rotation:
        rotation_yuv_file = bitrate_path + 'rotation_1_yuv.yuv'
        rotation_yuv_file2 = bitrate_path + 'rotation_2_yuv.yuv'
        rotation_qp = args.rot_qp
        rotation_yuv_mode = '444'
        rotation_yuv_mode2 = '444'
        print("\n--------hevc coding (rotation)-----------")
        rotation_yuv_lossy = 0
        rot1_bpp, rot1_enc_time, rot1_dec_time = HEVC_lossy(rotation_yuv_file, num_frames, height, width, bitrate_path, rotation_original_bd, rotation_qp, rotation_yuv_mode, HM_encoder, HM_decoder, HM_cfg, max_CU, CU_depth)   
        rot2_bpp, rot2_enc_time, rot2_dec_time = HEVC_lossy(rotation_yuv_file2, num_frames, height, width, bitrate_path, rotation_original_bd, rotation_qp, rotation_yuv_mode2, HM_encoder, HM_decoder, HM_cfg, max_CU, CU_depth)
        rec_image(rotation_yuv_file.replace("_yuv.yuv", "_de.yuv"), img_reconstruction_path, rotation_original_bd, width, height)
        rec_image(rotation_yuv_file2.replace("_yuv.yuv", "_de.yuv"), img_reconstruction_path, rotation_original_bd, width, height)
        rot_bpp = rot1_bpp + rot2_bpp
        rot_enc_time = rot1_enc_time + rot2_enc_time
        rot_dec_time = rot1_dec_time + rot2_dec_time

    # compress scaling
    if flag_scaling:
        scale_yuv_file = bitrate_path + 'scale_yuv.yuv'
        scale_qp = args.scale_qp
        scale_yuv_mode = '444'
        scale_yuv_lossy = 0
        print("\n--------hevc coding (scaling)-----------")
        scale_bpp, scale_enc_time, scale_dec_time = HEVC_lossy(scale_yuv_file, num_frames, height, width, bitrate_path, scale_original_bd, scale_qp, scale_yuv_mode, HM_encoder, HM_decoder, HM_cfg, max_CU, CU_depth)
        rec_image(scale_yuv_file.replace("_yuv.yuv", "_de.yuv"), img_reconstruction_path, scale_original_bd, width, height)
   
    if flag_xyz:
    # compress xyz
        xyz_1_yuv_file = bitrate_path + 'xyz_1_yuv.yuv'
        xyz_2_yuv_file = bitrate_path + 'xyz_2_yuv.yuv'
        xyz_yuv_mode = '444'
        xyz_yuv_loss = 0
        print("\n--------hevc coding (xyz)-----------")
        xyz_1_bpp, xyz_1_enc_time, xyz_1_dec_time = HEVC_lossless(xyz_1_yuv_file, num_frames, height, width, bitrate_path, xyz_MSB, xyz_yuv_mode, HM_encoder, HM_decoder, HM_cfg, max_CU, CU_depth)
        xyz_2_bpp, xyz_2_enc_time, xyz_2_dec_time = HEVC_lossless(xyz_2_yuv_file, num_frames, height, width, bitrate_path, xyz_LSB, xyz_yuv_mode, HM_encoder, HM_decoder, HM_cfg,  max_CU, CU_depth)
        rec_image(xyz_1_yuv_file.replace("_yuv.yuv", "_de.yuv"), img_reconstruction_path, xyz_MSB, width, height)
        rec_image(xyz_2_yuv_file.replace("_yuv.yuv", "_de.yuv"), img_reconstruction_path, xyz_LSB, width, height)
        xyz_bpp = xyz_1_bpp + xyz_2_bpp
        xyz_enc_time = xyz_1_enc_time + xyz_2_enc_time
        xyz_dec_time = xyz_1_dec_time + xyz_2_dec_time

    # recover gs file
    rec_time_list = []

    folder_name = f"{0:03d}"
    folder_path = os.path.join(img_reconstruction_path, folder_name)
    dec_image_file = natsorted(glob.glob(f"{folder_path}/*.png"))
    meta_data_file = sequence_path+frames + '/metadata.json'

    with open(meta_data_file, "r") as f:
        meta_data = json.load(f)
    PCA_dim = meta_data["AC_dim"]
    with open(sequence_path+frames + '/pca_AC_all.json', "r") as f:
        PCA_meta = json.load(f)
    rec_ply_path = os.path.join(reconstruction_path, folder_name)
    rec_time = rec_gs(dec_image_file, height, width, meta_data, rec_ply_path, PCA_meta)
    rec_time_list.append(rec_time)


    codec_name = 'HM18.0'
    if flag_xyz and flag_f_dc and flag_f_rest and flag_opacity and flag_rotation and flag_scaling:
        ic("Saving bpp.json...")
        all_bpp = xyz_bpp + f_dc_bpp + f_rest_file_size + opacity_bpp + scale_bpp + rot_bpp
        all_size = all_bpp * height * width /(1024**2)
        f_rest_size_sub = [(f_rest_subbpp* height * width /(1024**2)/num_frames) for f_rest_subbpp in f_rest_bpp]
        f_rest_size_all = f_rest_file_size * height * width /(1024**2)/num_frames
        f_rest_bpp_per = [(f*8/num_frames) for f in f_rest_bpp]
        bpp_file = bitrate_path + 'bpp.json'
        bpp_data = {
            "Codec": codec_name,
            "frames": num_frames,
            "xyz_bpp(bits per primitive)": xyz_bpp*8/num_frames,
            "xyz_MSB_bpp":xyz_1_bpp*8/num_frames,
            "xyz_LSB_bpp":xyz_2_bpp*8/num_frames,
            "f_dc_bpp": f_dc_bpp*8/num_frames,
            "f_rest_bpp":f_rest_bpp_per,
            "f_rest_bpp_all": sum(f_rest_bpp_per),
            "opacity_bpp": opacity_bpp*8/num_frames,
            "scale_bpp": scale_bpp*8/num_frames,
            "rot_bpp": rot_bpp*8/num_frames,
            "Overall_bpp": all_bpp*8/num_frames,

            "xyz_size": xyz_bpp* height * width /(1024**2)/num_frames,
            "xyz_MSB_size": xyz_1_bpp* height * width /(1024**2)/num_frames,
            "xyz_LSB_size": xyz_2_bpp* height * width /(1024**2)/num_frames,
            "f_dc_size": f_dc_bpp* height * width /(1024**2)/num_frames,
            "f_rest_size": f_rest_size_all,
            "opacity_size": opacity_bpp* height * width /(1024**2)/num_frames,
            "scale_size": scale_bpp* height * width /(1024**2)/num_frames,
            "rot_size": rot_bpp* height * width /(1024**2)/num_frames,
            "f_rest_size_sub": f_rest_size_sub,
            "Overall_size (MB)": all_size/num_frames,


            "xyz_enc_time": xyz_enc_time,
            "xyz_MSB_enc_time": xyz_1_enc_time,
            "xyz_LSB_enc_time": xyz_2_enc_time,
            "f_dc_enc_time": f_dc_enc_time,
            "f_rest_enc_time": f_rest_enc_time,
            "opacity_enc_time": opacity_enc_time,
            "scale_enc_time": scale_enc_time,
            "rot_enc_time": rot_enc_time,
            "Overall_enc_time": xyz_enc_time + f_dc_enc_time + f_rest_enc_time_all + opacity_enc_time + scale_enc_time + rot_enc_time,

            "xyz_dec_time": xyz_dec_time,
            "xyz_MSB__dec_time": xyz_1_dec_time,
            "xyz_LSB_dec_time": xyz_2_dec_time,
            "f_dc_dec_time": f_dc_dec_time,
            "f_rest_dec_time": f_rest_dec_time,
            "opacity_dec_time": opacity_dec_time,
            "scale_dec_time": scale_dec_time,
            "rot_dec_time": rot_dec_time,
            "Overall_dec_time": xyz_dec_time + f_dc_dec_time + f_rest_dec_time_all + opacity_dec_time + scale_dec_time + rot_dec_time,
            
            "Postprocessing_time (dequantization, 2DMapto3DGS)": rec_time_list
        }
        bpp_data = {k: float(v) if isinstance(v, np.floating) else v for k, v in bpp_data.items()}

        with open(bpp_file, "w") as f:
            json.dump(bpp_data, f, indent=4)
    
    bit_folder = Path(bitrate_path)
    for ext in ("*.yuv", "*.bin"):
        for file_path in bit_folder.glob(ext):
            try:
                file_path.unlink()
                print(f"delete: {file_path}")
            except Exception as e:
                print(f"delete failed for {file_path}: {e}")


