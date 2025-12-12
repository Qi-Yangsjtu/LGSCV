from argparse import ArgumentParser, Namespace
import sys
import os
filedir = os.path.dirname(os.path.abspath(__file__))
rootdir =os.path.dirname(filedir)


class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()

        # for key, value in vars(self).items():
        #     setattr(group, key, value)

        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group
    

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.data_device = "cuda:0"
        self.eval = True
        self.model_path = "./output/"
        self.train_test_exp = False
        self.bit_save_path = "./output/bit/"
        self.max_leaf_nodes = 8
        # self.max_leaf_nodes_color = 16
        self.geo_chunk_size = 2
        self.geo_BD = 16
        self.SH_cut = 0
        '''
        anchor model 0 means using some certain as anchor
        1 means using average
        '''
        self.anchor_model = 0
        '''
        anchor index = 0 means using the first, otherwise using the middle
        '''
        self.anchor_index = 0

        '''
        color quan type:
        0: dynamic for all
        1: fix all as 1
        2: res as 1, anchor and hp as dynamic
        '''
        self.color_quan_type = 1
        '''
        geo quan type:
        0: dynamic for all
        1: fix all as 1
        2: hp as dynamic, other 1
        3: anchor and res as dynamic, hp as 1
        '''
        self.geo_quan_type = 3
        self.geo_quan_type_single = 3

        self.geo_Q_ini = 0.01
        self.geo_Q_op = 0.08
        self.geo_Q_sc = 0.05
        self.geo_Q_ro = 0.007

        self.dual_color_res = 1
        self.dual_color_Q = 0.01

        self.fea_color_dec = [128, 64, 48],
        self.res_color_enc = [128, 64, 32],

        super().__init__(parser, "Loading Parameters", sentinel)

    # def extract(self, args):
    #     g = super().extract(args)
    #     g.source_path = os.path.abspath(g.source_path)
    #     return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 50
        self.iterations2 = 10
        
        self.color_iteration = 10
        self.geo_iteration = 10

        self.percent_dense = 0.001
        self.lambda_dssim = 0.2
        self.lambda_rd = 0.00005
        self.opacity_scale = 1000
        self.training_dataset = os.path.join(rootdir, "../GS_dataset/training_avs/")
        self.testing_dataset = os.path.join(rootdir, "../GS_dataset/testing_avs/")

        self.image_training_dataset = os.path.join(rootdir, "../GS_dataset/kodak/")
        self.image_testing_dataset = os.path.join(rootdir, "../GS_dataset/kodak_test/")
        
        self.training_record = os.path.join(rootdir, "../GS_dataset/DL3DV-10K_tiny/final_train_list_Ng_L40S.txt")
        self.testing_record = os.path.join(rootdir, "../GS_dataset/DL3DV-10K_tiny/final_test_list_ckpt_L40S.txt")
        
        self.training_record_all = os.path.join(rootdir, "../DL3DVFILE/final_train_list_Ng_L40S.txt")
        self.testing_record_all = os.path.join(rootdir, "../DL3DVFILE/final_test_list_ckpt_L40S.txt")
        
        
        self.training_record_sub = os.path.join(rootdir, "../DL3DVFILE_subset/final_train_list_Ng_L40S.txt")
        # self.testing_record_all = os.path.join(rootdir, "../DL3DVFILE/final_test_list_ckpt_L40S.txt")
        
        self.training_record_sub2 = os.path.join(rootdir, "../DL3DV/train_list.txt")
        # self.repclace_folder = os.path.join(rootdir, '../prune_tiny/')
        self.saving_iteration = [1, 2]
        # self.each_sample_iteration = 1
        self.testing_iteration = [1]
        self.color_saving_iteration = [1, 2]
        # self.each_sample_iteration = 1
        self.color_testing_iteration = [1]
        self.geo_saving_iteration = [1, 2]
        # self.each_sample_iteration = 1
        self.geo_testing_iteration = [1]
        self.remove_outlier_for_xyz = True
        self.camera_batch = 4
        self.model_lr = 1e-4
        self.model_minlr = 1e-6

        self.lr_decay_flag = True
        self.lr_decay_interval = 5
        self.joint_train_epoch = 50

        self.start_iteration = 1
        self.dynamic_geo_after_join = False
        self.geo_bit_weight = 1
        self.dynamic_lambda = 1
        self.dynamic_lambda_seg = 4

        self.training_dataset_class = 0
        # self.SH_cut = 0
        # self.single_geo = 10
        self.camera_num = 30
        self.factorized_epochs = 5
        self.stop_dylr = 20
        self.no_bpp_epoch = 5

        self.entropy_flag = 1
        
        
        self.dual_color_cha = 3
        self.dual_op_ratio = 0.2
        self.dual_color_GradientClip = 0
        self.load_model_path = ""
        self.dual_color_iteration = 60

        self.no_bpp_epoch = 10
        self.no_bpp = 0

        self.sh_rgb2yuv = 0
        
  
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)