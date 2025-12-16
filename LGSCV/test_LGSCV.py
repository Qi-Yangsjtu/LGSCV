from icecream import ic
import os


#**************************************************************
# This script to test LGSCV, Lightweight 3D Gaussian Splatting Compression via Video Codec (DCC2026 Oral)
# Author: Qi Yang, University of Missouri - Kansas City 
# Email: qiyang@umkc.edu
#**************************************************************

dataset = 'xxx/MPEG/' # set you own dataset path
sub_dataset = [ 'bartender'] # set you own sub-dataset path
sub_frame = f'frame{0:03d}' # set .ply name

# QP list for HM18.0 compression
# SH DC, SH AC, Opacity, Scale, and Rotation
f_dc_qp_list = [0, 7, 7, 7, 7]
f_rest_qp_1_list = [0, 7, 17, 22, 32]
opacity_qp_list = [0, 7, 17, 17, 22]
scale_qp_list = [0, 7, 7, 12, 12]
rot_qp_list = [0, 2, 2, 7, 17]

# Control parameters
PCA_dim = '12'   # PCA dimension for SH AC

mini_PLAS_size = '3' # MBS, maximum block size = 2*mini_PLAS_size + 2, for example, '3' -> MBS is '8*8'

single_scale_miniPLAS = 2 # block size module: 1 for single scale, e.g., [8], 0 for progressive [8, 4, ...], 2 for repeat, [8,8]

if single_scale_miniPLAS == 1:
    mini_PLAS_block_size  = str(2* int(mini_PLAS_size) + 2)
    miniPLAS_single_scale ='1'
elif single_scale_miniPLAS == 0:
    mini_PLAS_block_size = '4'
    miniPLAS_single_scale ='0'
else:
    mini_PLAS_block_size  = str(2* int(mini_PLAS_size) + 2)
    miniPLAS_single_scale ='2'

max_CU = '64'  # CU size of HM18.0
CU_depth = '4' # PD of HM18.0



######DO not Change Following Parameters, otherwise, the script will report some bugs.########################################
mini_PLAS = '1' # enable MiniPLAS
f_dc_clip = '0'  # whether using DC clipping before compression, in our paper, we set thsi flag as 1, however, using 0 will has higher quality (default and suggest)
I_mini_PLAS_size = mini_PLAS_size
radius_update_ratio = '0.5'
radius_iniSize_scale = '1'
I_mini_PLAS_block_size = mini_PLAS_block_size
I_miniPLAS_single_scale = miniPLAS_single_scale
#############################################################################################################################


results_main_path = f'/HEVC_test'

# better to use absolute path
map_script = 'GSTo2DMap.py'
coding_script = 'image_coding.py'
render_script = 'render.py'
metric_script = 'test_metric.py'

device = '0' # GPU index


flag_mapping = True
flag_coding = True
flag_testing = True
flag_statistic = True


#*************************************Convert 3D GS into 2D maps*************************************
if flag_mapping:
    for sub_scene in sub_dataset:
        input = dataset+sub_scene+'/'+sub_frame+'.ply'
        output = dataset+sub_scene+results_main_path
        mini_PLAS_size  = I_mini_PLAS_size
        mini_PLAS_block_size =   I_mini_PLAS_block_size
        map_result = output + '/2Dmap/'+sub_frame + '/'
        input_dq = map_result+'point_cloud.ply'
        map_log = map_result + '/log.txt'
        os.makedirs(os.path.dirname(map_log), exist_ok=True)
        cmd_map = f'CUDA_VISIBLE_DEVICES={device} python ' + map_script 
        cmd_map = cmd_map + ' --gaussian_ply='+input
        cmd_map = cmd_map + ' --result_path_save='+map_result 
        cmd_map = cmd_map + ' --radius_update_ratio='+radius_update_ratio
        cmd_map = cmd_map + ' --PCAdim_for_AC='+PCA_dim
        cmd_map = cmd_map + ' --mini_PLAS='+mini_PLAS
        cmd_map = cmd_map + ' --mini_PLAS_size='+mini_PLAS_size
        cmd_map = cmd_map + ' --mini_PLAS_block_size='+mini_PLAS_block_size
        cmd_map = cmd_map + ' --mini_PLAS_single_scale='+miniPLAS_single_scale
        cmd_map = cmd_map +  f' >{map_log} 2>&1 '
        ic(cmd_map)
        os.system(cmd_map)

#**********************************Coding with HM18.0************************************************
if flag_coding:
    for sub_scene in sub_dataset:
        output = dataset+sub_scene+results_main_path      
        for i in range(len(f_rest_qp_1_list)):
            sequence_path = output + f'/2Dmap/'
            f_dc_qp = str(f_dc_qp_list[i])
            f_rest_1_qp = str(f_rest_qp_1_list[i])
            opacity_qp = str(opacity_qp_list[i])
            scale_qp = str(scale_qp_list[i])
            rot_qp = str(rot_qp_list[i])
            coding_bitrate_path = output +'/bitrate_'+str(i)+'/'
            coding_reconstruction_path = output +'/bitrate_'+str(i)+'_reply/'
            Image_reconstruction_path = output +'/bitrate_'+str(i)+'_reImg/'
            coding_log = coding_bitrate_path + 'log.txt'
            os.makedirs(os.path.dirname(coding_log), exist_ok=True)
            coding_cmd = f'CUDA_VISIBLE_DEVICES={device} python ' + coding_script
            coding_cmd = coding_cmd + ' --sequence_path='+sequence_path
            coding_cmd = coding_cmd + ' --bitrate_path='+coding_bitrate_path
            coding_cmd = coding_cmd + ' --reconstruction_path='+coding_reconstruction_path
            coding_cmd = coding_cmd + ' --reconstruction_img_path='+Image_reconstruction_path
            coding_cmd = coding_cmd + ' --f_dc_qp='+f_dc_qp
            coding_cmd = coding_cmd + ' --f_rest_1_qp='+f_rest_1_qp
            coding_cmd = coding_cmd + ' --opacity_qp='+opacity_qp
            coding_cmd = coding_cmd + ' --scale_qp='+scale_qp
            coding_cmd = coding_cmd + ' --rot_qp='+rot_qp
            coding_cmd = coding_cmd + ' --max_CU='+max_CU
            coding_cmd = coding_cmd + ' --CU_depth='+CU_depth
            coding_cmd = coding_cmd + f' >{coding_log} 2>&1 '
            ic(coding_cmd)
            os.system(coding_cmd)




#**********************************rendering and test metric*****************************************************
if flag_testing:
    for sub_scene in sub_dataset:
        output = dataset+sub_scene+results_main_path
        index = 0
        input = dataset+sub_scene+'/'+sub_frame+'.ply'
        for i in range(len(f_rest_qp_1_list)):

            map_result = output + f'/2Dmap/'+sub_frame + '/'
            camera_json= dataset+sub_scene+'/cameras.json'
            ref_file = input
            coding_reconstruction_path = output +'/bitrate_'+str(i)+'_reply/'
            dis_file = coding_reconstruction_path + f'{index:03d}/point_cloud.ply'
            rendering_path = output + f'/render/'
            ref_path = rendering_path + f'ref_render/{index:03d}/'
            dis_path = rendering_path + 'dis_render_'+str(i)+f'/{index:03d}/'
            rendering_dis_log = dis_path + '/log.txt'
           
            os.makedirs(os.path.dirname(rendering_dis_log), exist_ok=True)
            if sub_scene =='bartender' or sub_scene =='cinema':
                test_camera = [9, 11]
            else: # for 'breakfast'
                test_camera = [6, 8]
            
            if i ==0:
                rendering_ref_log = ref_path + '/log.txt'
                os.makedirs(os.path.dirname(rendering_ref_log), exist_ok=True)
                render_ref_cmd = f'CUDA_VISIBLE_DEVICES={device} python '+ render_script
                render_ref_cmd = render_ref_cmd + ' --camera_json='+camera_json
                render_ref_cmd = render_ref_cmd + ' --model='+ref_file
                render_ref_cmd = render_ref_cmd + ' --result_root='+ref_path
                render_ref_cmd = render_ref_cmd + ' --test_camera '+ ' '.join(map(str, test_camera))
                render_ref_cmd = render_ref_cmd + f' >{rendering_ref_log} 2>&1 '
                ic(render_ref_cmd)
                os.system(render_ref_cmd)


            render_dis_cmd = f'CUDA_VISIBLE_DEVICES={device} python '+ render_script
            render_dis_cmd = render_dis_cmd + ' --camera_json='+camera_json
            render_dis_cmd = render_dis_cmd + ' --model='+dis_file
            render_dis_cmd = render_dis_cmd + ' --result_root='+dis_path
            render_dis_cmd = render_dis_cmd + ' --test_camera '+ ' '.join(map(str, test_camera))
            render_dis_cmd = render_dis_cmd + f' >{rendering_dis_log} 2>&1 '
            ic(render_dis_cmd)
            os.system(render_dis_cmd)

            metric_path = rendering_path + f'dis_render_'+str(i)+f'_metric/{index:03d}/'
            metric_log = metric_path + '/log.txt'
            os.makedirs(os.path.dirname(metric_log), exist_ok=True)
            metric_cmd = f'CUDA_VISIBLE_DEVICES={device} python '+ metric_script
            metric_cmd = metric_cmd + ' --ref_img='+ref_path
            metric_cmd = metric_cmd + ' --dis_img='+dis_path
            metric_cmd = metric_cmd + ' --result_path='+metric_path
            metric_cmd = metric_cmd + f' >{metric_log} 2>&1 '
            ic(metric_cmd)
            os.system(metric_cmd)

        
        index+=1






if flag_statistic:
    import json
    import sys
    for sub_scene in sub_dataset:
        output = dataset+sub_scene+results_main_path+'/render/'
        for i in range(len(f_dc_qp_list)):
            dis_sequence_result = output + f'dis_render_{i}_metric/'
            psnr = []
            ssim = []
            lpips = []

            sub_frame_results = dis_sequence_result + f'{0:03d}/'
            score_file = sub_frame_results + 'Overall_Metric.json'
            with open(score_file, "r") as f:
                data = json.load(f)

            for k, v in data.items():
                psnr.append(v.get("PSNR"))
                ssim.append(v.get("SSIM"))
                lpips.append(v.get("LPIPS"))
            
            psnr_ave = sum(psnr)/len(psnr)
            ssim_ave = sum(ssim)/len(ssim)
            lpips_ave = sum(lpips)/len(lpips)
            ave_data = {
                'PSNR_ave': psnr_ave,
                'SSIM_ave': ssim_ave,
                'LPIPS_ave': lpips_ave,
            }
            file_save = dis_sequence_result + 'Metric_ave.json'
            with open(file_save, "w") as f:
                json.dump(ave_data, f, indent=4)
        


