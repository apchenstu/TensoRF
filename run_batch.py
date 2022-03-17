import os

cuda = 0
batch_size = 4096
N_vis = -1



########################   nerf   ##################################
# n_iters = 30000
# for data_name in ['mic']:#'ship','mic','chair','lego','drums','ficus','materials','hotdog'
#     cmd = f'CUDA_VISIBLE_DEVICES={cuda}  python train.py ' \
#           f'--dataset_name blender --datadir /mnt/new_disk_2/anpei/Dataset/nerf_synthetic/{data_name} '\
#           f'--expname {data_name}   --batch_size {batch_size} ' \
#           f'--n_iters {n_iters}  ' \
#           f'--N_voxel_init {128**3} --N_voxel_final {300**3} '\
#           f'--N_vis {5}  --vis_every {10000} ' \
#           f'--n_lamb_sigma "[16,16,16]" --n_lamb_sh "[48,48,48]" ' \
#           f'--upsamp_list "[2000,3000,4000,5500,7000]" --update_AlphaMask_list "[2000,4000]" ' \
#           f'--shadingMode MLP_Fea --fea2denseAct softplus  --view_pe {2} --fea_pe {2} ' \
#           f'--L1_weight_inital {8e-5} --L1_weight_rest {4e-5} --rm_weight_mask_thre {1e-4} --add_timestamp 0 ' \
#           f'--render_test 1 ' \
#           f'--render_only 1 --ckpt /mnt/new_disk_2/anpei/code/TensoRF/log/{data_name}.th '
#     print(cmd)
#     os.system(cmd)

########################   nsvf   ##################################
# n_iters = 30000
# for data_name in ['Robot']:#'Bike','Lifestyle','Palace','Robot','Spaceship','Steamtrain','Toad','Wineholder'
#     cmd = f'CUDA_VISIBLE_DEVICES={cuda}  python train.py ' \
#           f'--dataset_name nsvf --datadir /mnt/new_disk_2/anpei/Dataset/TeRF/Synthetic_NSVF/{data_name} '\
#           f'--expname {data_name} --batch_size {batch_size} ' \
#           f'--n_iters {n_iters}  ' \
#           f'--N_voxel_init {128**3} --N_voxel_final {300**3} '\
#           f'--N_vis {5}  ' \
#           f'--n_lamb_sigma "[16,16,16]" --n_lamb_sh "[48,48,48]" ' \
#           f'--upsamp_list "[2000, 3000, 4000, 5500,7000]" --update_AlphaMask_list "[2000,4000]" ' \
#           f'--shadingMode MLP_Fea --fea2denseAct softplus  --view_pe {2} --fea_pe {2} ' \
#           f'--L1_weight_inital {8e-5} --L1_weight_rest {4e-5} --rm_weight_mask_thre {1e-4} --add_timestamp 0 ' \
#           f'--render_test 1 ' \
#           f'--render_only 1 --ckpt /mnt/new_disk_2/anpei/code/TensoRF/log/{data_name}.th '
#     print(cmd)
#     os.system(cmd)

########################   tankstemple   ##################################
# n_iters = 30000
# for data_name in ['Truck']:#'Truck','Barn','Caterpillar','Family','Ignatius'
#     cmd = f'CUDA_VISIBLE_DEVICES={cuda}  python train.py ' \
#           f'--dataset_name tankstemple --datadir /mnt/new_disk_2/anpei/Dataset/TeRF/TanksAndTemple//{data_name} '\
#           f'--expname {data_name} --batch_size {batch_size} ' \
#           f'--n_iters {n_iters}  ' \
#           f'--N_voxel_init {128**3} --N_voxel_final {300**3} '\
#           f'--N_vis {5} ' \
#           f'--n_lamb_sigma "[16,16,16]" --n_lamb_sh "[48,48,48]" ' \
#           f'--upsamp_list "[2000,3000,4000,5500,7000]" --update_AlphaMask_list "[2000,4000]" ' \
#           f'--shadingMode MLP_Fea --fea2denseAct softplus  --view_pe {2} --fea_pe {2} ' \
#           f'--TV_weight_density {0.1} --TV_weight_app {0.01}  --add_timestamp 0 ' \
#           f'--render_path 1  ' \
#           f'--render_only 1 --ckpt /mnt/new_disk_2/anpei/code/TensoRF/log/{data_name}.th'
#     print(cmd)
#     os.system(cmd)

#########################   llff   ##################################
n_iters = 5000
for data_name in ['trex']:# 'room','fortress', 'flower','orchids','leaves','horns','trex','fern'
    cmd = f'CUDA_VISIBLE_DEVICES={cuda}  python train.py ' \
          f'--dataset_name llff --datadir /mnt/new_disk_2/anpei/Dataset/MVSNeRF/nerf_llff_data/{data_name} --downsample_train 4.0 --ndc_ray 1 '\
          f'--expname {data_name}   --batch_size {batch_size} ' \
          f'--n_iters {n_iters}  ' \
          f'--N_voxel_init {128**3} --N_voxel_final {640**3} '\
          f'--N_vis {N_vis}  ' \
          f'--n_lamb_sigma "[16,4,4]" --n_lamb_sh "[48,12,12]" ' \
          f'--upsamp_list "[2000,3000,4000,5500]" --update_AlphaMask_list "[2500]" ' \
          f'--shadingMode MLP_Fea --fea2denseAct relu --nSamples {512} --view_pe {0} --fea_pe {0} --lr_decay_iters 30000 ' \
          f'--L1_weight_inital {0.0}  --TV_weight_density {1.0} --TV_weight_app {1.0} --add_timestamp 0 ' \
          f'--render_only 1 --render_path 1 --ckpt /mnt/new_disk_2/anpei/code/TensoRF/log/trex.th'

    print(cmd)
    os.system(cmd)
