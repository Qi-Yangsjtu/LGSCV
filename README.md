# LGSCV: Lightweight 3D Gaussian Splatting Compression via Video Codec (DCC2026 Oral)

Qi Yang, Geert Van Der Auwera, Zhu Li <br>

[Paper](https://www.arxiv.org/abs/2512.11186) [PPT](https://drive.google.com/file/d/1JN4x5z1njUzunMHQJkSZe7BYYSmEaTgB/view?usp=sharing)

The rendering part of this project is based on the official implementation associated with the paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering", which can be found [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

The MiniPLAS part of this project is based on the official implementation associated with the paper "Compact 3D Scene Representation via Self-Organizing Gaussian Grids", which can be found [here](https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/)

Abstract: *Current video-based GS compression methods rely on using Parallel Linear Assignment Sorting (PLAS) to convert 3D GS into smooth 2D maps, which are computationally expensive and time-consuming, limiting the application of GS on lightweight devices. In this paper, we propose a Lightweight 3D Gaussian Splatting (GS) Compression method based on Video codec (LGSCV). First, a two-stage Morton scan is proposed to generate blockwise 2D maps that are friendly for canonical video codecs in which the coding units (CU) are square blocks. A 3D Morton scan is used to permute GS primitives, followed by a 2D Morton scan to map the ordered GS primitives to 2D maps in a blockwise style. However, although the blockwise 2D maps report close performance to the PLAS map in high-bitrate regions, they show a quality collapse at medium-to-low bitrates. Therefore, a principal component analysis (PCA) is used to reduce the dimensionality of spherical harmonics (SH), and a MiniPLAS, which is flexible and fast, is designed to permute the primitives within certain block sizes. Incorporating SH PCA and MiniPLAS leads to a significant gain in rate–distortion (RD) performance, especially at medium and low bitrates. MiniPLAS can also guide the setting of the codec CU size configuration and significantly reduce encoding time. Experimental results on the MPEG dataset demonstrate that the proposed LGSCV achieves over 20% RD gain compared with state-of-the-art methods, while reducing 2D map generation time to approximately 1 second and cutting encoding time by 50%. *


### Setup

#### Local Setup

Our default, provided install method is based on the Conda package and environment management:
```shell
unzip submodules.zip
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate LGSCV
```

### Running

To run the LGSCV, check ./LGSCV/test_LGSCV.py

You will obtain several bitrate results after running.

The results folder is like the following: 

-HEVC_test

    -2Dmap
        - frame000
            - config.json (Parameters of using GSTo2DMap.py)
            - log.txt (log information of running GSTo2DMap.py)
            - metadata.json (metadata for decoding)
            - time.json (computation time of running GSTo2DMap.py)
            - pca_AC_all.json (metadata for PCA)
            - xxx.png (2D images of input GS file)
    -bitrate_xx
        - bpp.json (bitrate details, encoding and decoding time)
        - config.json (Parameters for using image_coding.py)
        - log.txt (log information if running image_coding.py)
    -bitrate_xx_reImg (decompressed image of input 2D maps)
    -bitrate_xx_reply (lossy compressed GS file)
    -render
        - dis_render_xx (rendering results of bitrate xx)
        - dis_render_xx_metric (metric results (PSNR, SSIM, LPIPS) of bitrate xx)
        - ref_render (rendering results of input GS file)

Our project uses HM18.0, and we provide a Linux binary and cfg files in ./HM
You can compile your own HM binary via the project [here](https://vcgit.hhi.fraunhofer.de/jvet/HM/-/tree/HM-18.0?ref_type=tags), and update the binary path in image_coding.py



If you use our project, please cite our paper.
