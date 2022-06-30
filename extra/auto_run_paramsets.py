import os
import threading, queue
import numpy as np
import time


def getFolderLocker(logFolder):
    while True:
        try:
            os.makedirs(logFolder+"/lockFolder")
            break
        except: 
            time.sleep(0.01)

def releaseFolderLocker(logFolder):
    os.removedirs(logFolder+"/lockFolder")

def getStopFolder(logFolder):
    return os.path.isdir(logFolder+"/stopFolder")


def get_param_str(key, val):
    if key == 'data_name':
        return f'--datadir {datafolder}/{val} '
    else:
        return f'--{key} {val} '

def get_param_list(param_dict):
    param_keys = list(param_dict.keys())
    param_modes = len(param_keys)
    param_nums = [len(param_dict[key]) for key in param_keys]
    
    param_ids = np.zeros(param_nums+[param_modes], dtype=int)
    for i in range(param_modes):
        broad_tuple = np.ones(param_modes, dtype=int).tolist()
        broad_tuple[i] = param_nums[i]
        broad_tuple = tuple(broad_tuple)
        print(broad_tuple)
        param_ids[...,i] = np.arange(param_nums[i]).reshape(broad_tuple)
    param_ids = param_ids.reshape(-1, param_modes)
    # print(param_ids)
    print(len(param_ids))
    
    params = []
    expnames = []
    for i in range(param_ids.shape[0]):
        one = ""
        name = ""
        param_id = param_ids[i]
        for j in range(param_modes):
            key = param_keys[j]
            val = param_dict[key][param_id[j]]
            if type(key) is tuple:
                assert len(key) == len(val)
                for k in range(len(key)):
                    one += get_param_str(key[k], val[k])
                    name += f'{val[k]},'
                name=name[:-1]+'-'
            else:
                one += get_param_str(key, val)
                name += f'{val}-'
        params.append(one)
        name=name.replace(' ','')
        print(name)
        expnames.append(name[:-1])
    # print(params)
    return params, expnames







if __name__ == '__main__':
    


    # nerf
    expFolder = "nerf/"
    # parameters to iterate, use tuple to couple multiple parameters
    datafolder = '/mnt/new_disk_2/anpei/Dataset/nerf_synthetic/'
    param_dict = {
        'data_name': ['ship', 'mic', 'chair', 'lego', 'drums', 'ficus', 'hotdog', 'materials'],
        'data_dim_color': [13, 27, 54]
    }

    # n_iters = 30000
    # for data_name in ['Robot']:#'Bike','Lifestyle','Palace','Robot','Spaceship','Steamtrain','Toad','Wineholder'
    #     cmd = f'CUDA_VISIBLE_DEVICES={cuda}  python train.py ' \
    #           f'--dataset_name nsvf --datadir /mnt/new_disk_2/anpei/Dataset/TeRF/Synthetic_NSVF/{data_name} '\
    #           f'--expname {data_name} --batch_size {batch_size} ' \
    #           f'--n_iters {n_iters}  ' \
    #           f'--N_voxel_init {128**3} --N_voxel_final {300**3} '\
    #           f'--N_vis {5}  ' \
    #           f'--n_lamb_sigma "[16,16,16]" --n_lamb_sh "[48,48,48]" ' \
    #           f'--upsamp_list "[2000, 3000, 4000, 5500,7000]" --update_AlphaMask_list "[3000,4000]" ' \
    #           f'--shadingMode MLP_Fea --fea2denseAct softplus  --view_pe {2} --fea_pe {2} ' \
    #           f'--L1_weight_inital {8e-5} --L1_weight_rest {4e-5} --rm_weight_mask_thre {1e-4} --add_timestamp 0 ' \
    #           f'--render_test 1 '
    #     print(cmd)
    #     os.system(cmd)

    # nsvf
    # expFolder = "nsvf_0227/"
    # datafolder = '/mnt/new_disk_2/anpei/Dataset/TeRF/Synthetic_NSVF/'
    # param_dict = {
    #             'data_name': ['Robot','Steamtrain','Bike','Lifestyle','Palace','Spaceship','Toad','Wineholder'],#'Bike','Lifestyle','Palace','Robot','Spaceship','Steamtrain','Toad','Wineholder'
    #             'shadingMode': ['SH'],
    #             ('n_lamb_sigma', 'n_lamb_sh'): [ ("[8,8,8]", "[8,8,8]")],
    #             ('view_pe', 'fea_pe', 'featureC','fea2denseAct','N_voxel_init') : [(2, 2, 128, 'softplus',128**3)],
    #             ('L1_weight_inital', 'L1_weight_rest', 'rm_weight_mask_thre'):[(4e-5, 4e-5, 1e-4)],
    #             ('n_iters','N_voxel_final'): [(30000,300**3)],
    #             ('dataset_name','N_vis','render_test') : [("nsvf",5,1)],
    #             ('upsamp_list','update_AlphaMask_list'): [("[2000,3000,4000,5500,7000]","[3000,4000]")]
    #
    #     }

    # tankstemple
    # expFolder = "tankstemple_0304/"
    # datafolder = '/mnt/new_disk_2/anpei/Dataset/TeRF/TanksAndTemple/'
    # param_dict = {
    #             'data_name': ['Truck','Barn','Caterpillar','Family','Ignatius'],
    #             'shadingMode': ['MLP_Fea'],
    #             ('n_lamb_sigma', 'n_lamb_sh'): [("[16,16,16]", "[48,48,48]")],
    #             ('view_pe', 'fea_pe','fea2denseAct','N_voxel_init','render_test') : [(2, 2, 'softplus',128**3,1)],
    #             ('TV_weight_density','TV_weight_app'):[(0.1,0.01)],
    #             # ('L1_weight_inital', 'L1_weight_rest', 'rm_weight_mask_thre'): [(4e-5, 4e-5, 1e-4)],
    #             ('n_iters','N_voxel_final'): [(15000,300**3)],
    #             ('dataset_name','N_vis') : [("tankstemple",5)],
    #             ('upsamp_list','update_AlphaMask_list'): [("[2000,3000,4000,5500,7000]","[2000,4000]")]
    #     }

    # llff
    # expFolder = "real_iconic/"
    # datafolder = '/mnt/new_disk_2/anpei/Dataset/MVSNeRF/real_iconic/'
    # List = os.listdir(datafolder)
    # param_dict = {
    #             'data_name': List,
    #             ('shadingMode', 'view_pe', 'fea_pe','fea2denseAct', 'nSamples','N_voxel_init') : [('MLP_Fea', 0, 0, 'relu',512,128**3)],
    #             ('n_lamb_sigma', 'n_lamb_sh') : [("[16,4,4]", "[48,12,12]")],
    #             ('TV_weight_density', 'TV_weight_app'):[(1.0,1.0)],
    #             ('n_iters','N_voxel_final'): [(25000,640**3)],
    #             ('dataset_name','downsample_train','ndc_ray','N_vis','render_path') : [("llff",4.0, 1,-1,1)],
    #             ('upsamp_list','update_AlphaMask_list'): [("[2000,3000,4000,5500,7000]","[2500]")],
    #     }

    # expFolder = "llff/"
    # datafolder = '/mnt/new_disk_2/anpei/Dataset/MVSNeRF/nerf_llff_data'
    # param_dict = {
    #             'data_name': ['fern', 'flower', 'room', 'leaves', 'horns', 'trex', 'fortress', 'orchids'],#'fern', 'flower', 'room', 'leaves', 'horns', 'trex', 'fortress', 'orchids'
    #             ('n_lamb_sigma', 'n_lamb_sh'): [("[16,4,4]", "[48,12,12]")],
    #             ('shadingMode', 'view_pe', 'fea_pe', 'featureC','fea2denseAct', 'nSamples','N_voxel_init') : [('MLP_Fea', 0, 0, 128, 'relu',512,128**3),('SH', 0, 0, 128, 'relu',512,128**3)],
    #             ('TV_weight_density', 'TV_weight_app'):[(1.0,1.0)],
    #             ('n_iters','N_voxel_final'): [(25000,640**3)],
    #             ('dataset_name','downsample_train','ndc_ray','N_vis','render_test','render_path') : [("llff",4.0, 1,-1,1,1)],
    #             ('upsamp_list','update_AlphaMask_list'): [("[2000,3000,4000,5500,7000]","[2500]")],
    #     }

    #setting available gpus
    gpus_que = queue.Queue(3)
    for i in [1,2,3]:
        gpus_que.put(i)
    
    os.makedirs(f"log/{expFolder}", exist_ok=True)

    def run_program(gpu, expname, param):
        cmd = f'CUDA_VISIBLE_DEVICES={gpu}  python train.py ' \
            f'--expname {expname} --basedir ./log/{expFolder} --config configs/lego.txt ' \
            f'{param}' \
            f'> "log/{expFolder}{expname}/{expname}.txt"'
        print(cmd)
        os.system(cmd)
        gpus_que.put(gpu)

    params, expnames = get_param_list(param_dict)

    
    logFolder=f"log/{expFolder}"
    os.makedirs(logFolder, exist_ok=True)

    ths = []
    for i in range(len(params)):

        if getStopFolder(logFolder):
            break


        targetFolder = f"log/{expFolder}{expnames[i]}"
        gpu = gpus_que.get()
        getFolderLocker(logFolder)
        if os.path.isdir(targetFolder):
            releaseFolderLocker(logFolder)
            gpus_que.put(gpu)
            continue
        else:
            os.makedirs(targetFolder, exist_ok=True)
            print("making",targetFolder, "running",expnames[i], params[i])
            releaseFolderLocker(logFolder)


        t = threading.Thread(target=run_program, args=(gpu, expnames[i], params[i]), daemon=True)
        t.start()
        ths.append(t)
    
    for th in ths:
        th.join()