import os
import glob
import time
import argparse
import subprocess
import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import ants
from ants.utils.bias_correction import n4_bias_field_correction
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import knn_points

from seg.unet import UNet
from surface.net import SurfDeform
from sphere.net.sunet import SphereDeform
from sphere.net.utils import get_neighs_order
from sphere.net.loss import (
    edge_distortion,
    area_distortion)

from utils.mesh import (
    apply_affine_mat,
    taubin_smooth)

from utils.register import (
    registration,
    ants_trans_to_mat)

from utils.io import (
    Logging,
    save_numpy_to_nifti,
    save_gifti_surface,
    save_gifti_metric,
    create_wb_spec)

from utils.inflate import (
    generate_inflated_surfaces,
    wb_generate_inflated_surfaces)

from utils.metric import (
    metric_dilation,
    cortical_thickness,
    curvature,
    sulcal_depth,
    myelin_map,
    smooth_myelin_map)


def chamfer_with_match_count(pred, gt):
    """
    Inputs:
    - pred: (B, N, 3) 预测点集
    - gt: (B, M, 3) 真实点集
    Returns:
    - chamfer_loss: scalar
    - match_counts: list[(M,) tensor] 每个 batch 中真实点被匹配的次数
    """

    # 若输入是 Pointclouds 对象，则转换为 padded tensor
    if isinstance(pred, Pointclouds):
        pred = pred.points_padded()
    if isinstance(gt, Pointclouds):
        gt = gt.points_padded()

    # 计算 Chamfer 距离（原始 loss）
    chamfer_loss, _ = chamfer_distance(pred, gt)

    # 用 KNN 搜索 pred->gt 最近邻，取最近的1个点
    knn_pred_to_gt = knn_points(pred, gt, K=1)

    # 获得匹配到的 gt 中的索引，形状为 (B, N, 1)
    indices = knn_pred_to_gt.idx[..., 0]  # 去掉最后一维 → (B, N)

    # 对每个 batch 中的 gt 点统计被匹配次数
    B, N = indices.shape
    M = gt.shape[1]
    match_counts = []
    for b in range(B):
        counts = torch.bincount(indices[b], minlength=M)
        match_counts.append(counts)

    return chamfer_loss, match_counts

# ============ load hyperparameters ============
parser = argparse.ArgumentParser(description="dHCP DL Neonatal Pipeline")
# parser.add_argument('--in_dir', default='/root/autodl-tmp/datasets/', type=str,
#                     help='Diectory containing input images.')
# parser.add_argument('--out_dir', default='/root/autodl-tmp/output/test/', type=str,
#                     help='Directory for saving the output of the pipeline.')
parser.add_argument('--in_dir', default='/root/autodl-tmp/hcp1200/sample_test/input/', type=str, # TODO
                    help='Diectory containing input images.')
parser.add_argument('--out_dir', default='/root/autodl-tmp/hcp1200/sample_test/output/', type=str, # TODO
                    help='Directory for saving the output of the pipeline.')
parser.add_argument('--T2', default='T2w_proc_affine.nii.gz', type=str,
                    help='Suffix of T2 image file.')
parser.add_argument('--T1', default='T1w_proc_affine.nii.gz', type=str,
                    help='Suffix of T1 image file.')
parser.add_argument('--sphere_proj', default='fs', type=str,
                    help='The method of spherical projection: [fs, mds].')
parser.add_argument('--device', default='cuda', type=str, # TODO
                    help='Device for running the pipeline: [cuda, cpu]')
parser.add_argument('--verbose', action='store_true',
                    help='Print debugging information.')
args = parser.parse_args()

in_dir = args.in_dir
out_dir = args.out_dir
t2_suffix = args.T2
t1_suffix = args.T1
sphere_type = args.sphere_proj
device = args.device
verbose = args.verbose
max_regist_iter = 5
min_regist_dice = 0.9


# ============ load nn model ============
# brain extraction
nn_seg_brain = UNet(
    C_in=1, C_hid=[16,32,32,32,32], C_out=1).to(device)
# ribbon segmentation
# nn_seg_ribbon = UNet(
#     C_in=1, C_hid=[16,32,64,128,128], C_out=1).to(device)

nn_seg_brain.load_state_dict( # done
    torch.load('/root/autodl-tmp/hcp1200/result_train_model/model_seg_brain_all.pt', map_location=device))
# nn_seg_ribbon.load_state_dict( # ignore
#     torch.load('./seg/model/model_seg_ribbon.pt', map_location=device))

# surface reconstruction
nn_surf_left_wm = SurfDeform( # done
    C_hid=[8,16,32,64,128,128], C_in=2, inshape=[160,304,256], sigma=1.0, device=device) # TODO
nn_surf_right_wm = SurfDeform( # done
    C_hid=[8,16,32,64,128,128], C_in=2, inshape=[160,304,256], sigma=1.0, device=device)
nn_surf_left_pial = SurfDeform( # done
    C_hid=[8,16,32,32,32,32], C_in=2, inshape=[160,304,256], sigma=1.0, device=device)
nn_surf_right_pial = SurfDeform( # done
    C_hid=[8,16,32,32,32,32], C_in=1, inshape=[160,304,256], sigma=1.0, device=device)

nn_surf_left_wm.load_state_dict( # training
    torch.load('/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_monai_01/model-0.1_hemi-left_wm_0003_230epochs.pt', map_location=device))
nn_surf_right_wm.load_state_dict(
    torch.load('/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_5_1/model_hemi-right_wm_0001_290epochs.pt', map_location=device))
nn_surf_left_pial.load_state_dict(
    torch.load('/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_5_1_pial/model_hemi_nc_-left_pial_0001_90epochs.pt', map_location=device))
nn_surf_right_pial.load_state_dict(
    torch.load('./surface/model/model_hemi-right_pial.pt', map_location=device))

img_t1_atlas_ants = ants.image_read('/root/autodl-tmp/hcp1200/template_hcp1200/MNI152_T1_0.7mm_brain_sampled.nii.gz')
affine_t1_atlas = nib.load('/root/autodl-tmp/hcp1200/template_hcp1200/MNI152_T1_0.7mm_brain_sampled.nii.gz').affine  # type: ignore

# ============ load input surface ============
surf_left_in = nib.load( # type: ignore
    '/root/autodl-tmp/hcp1200/template_hcp1200/lh.white.surf.gii')
# surf_left_in = nib.load( # type: ignore
#     '/root/dhcp/hcp_template/S1200.L.inflated_MSMAll.32k_fs_LR_150k.surf.gii')
vert_left_in = surf_left_in.agg_data('pointset') # type: ignore
face_left_in = surf_left_in.agg_data('triangle') # type: ignore
vert_left_in = apply_affine_mat(
    vert_left_in, np.linalg.inv(affine_t1_atlas))
vert_left_in = vert_left_in - [96,0,0]
face_left_in = face_left_in[:,[2,1,0]]
vert_left_in = torch.Tensor(vert_left_in[None]).to(device)          
face_left_in = torch.LongTensor(face_left_in[None]).to(device)

surf_right_in = nib.load( # type: ignore
    '/root/autodl-tmp/hcp1200/template_hcp1200/rh.white.surf.gii')
# surf_right_in = nib.load( # type: ignore
#     '/root/dhcp/hcp_template/S1200.R.inflated_MSMAll.32k_fs_LR_150k.surf.gii')
vert_right_in = surf_right_in.agg_data('pointset') # type: ignore
face_right_in = surf_right_in.agg_data('triangle') # type: ignore
vert_right_in = apply_affine_mat(
    vert_right_in, np.linalg.inv(affine_t1_atlas))
face_right_in = face_right_in[:,[2,1,0]]
vert_right_in = torch.Tensor(vert_right_in[None]).to(device)
face_right_in = torch.LongTensor(face_right_in[None]).to(device)



# ============ HCP DL-based neonatal pipeline ============
if __name__ == '__main__':
    total_surf_time = 0.0
    total_feature_time = 0.0
    total_inflate_time = 0.0
    num_subjects = 0
    t_surf = 0
    t_feature = 0
    t_inflate = 0
    chamfer_losses = []
    repeat_losses = []
    start_time = time.time()
    subj_list = sorted(glob.glob(in_dir + '**/*' + t2_suffix, recursive=True))
    for subj_t2_dir in tqdm(subj_list): # switch to t2

        total_surf_time += t_surf
        total_feature_time += t_feature
        total_inflate_time += t_inflate
        num_subjects += 1

        subj_id = subj_t2_dir.split('/')[-2] # switch to t2

        subj_out_dir = out_dir + subj_id + '/'
        # create output directory
        if not os.path.exists(subj_out_dir):
            os.mkdir(subj_out_dir)
            # add subject id as prefix
        subj_out_dir = subj_out_dir + subj_id
        
        # initialize logger
        logger = Logging(subj_out_dir)
        # start processing
        logger.info('========================================')
        logger.info('Start processing subject: {}'.format(subj_id))
        t_start = time.time()
 
        # ------ 惰性加载体积数据（T2w + T1w）------
        t2_img = nib.load(subj_t2_dir)
        affine_in = t2_img.affine
        t2_data = t2_img.get_fdata()
        t2_data = (t2_data / t2_data.max()).astype(np.float32)

        subj_t1_dir = subj_t2_dir.replace('T2w', 'T1w')  # 或根据路径自己拼
        t1_img = nib.load(subj_t1_dir)
        # t1_img = nib.load(os.path.join(subj_dir, 'T1w_proc_affine.nii.gz'))
        t1_data = t1_img.get_fdata()
        t1_data = (t1_data / t1_data.max()).astype(np.float32)

        # 合并为 2 通道体积，并调整维度顺序
        vol_data = np.stack([t1_data, t2_data], axis=0)  # [2, H, W, D]
        vol_in = torch.tensor(vol_data).unsqueeze(0).float().to(device)  # [1, 2, H, W, D]

        # surf_file = os.path.join(in_dir, subj_id, subj_id + '.L.white.native_160k.surf.gii')
        # if os.path.exists(surf_file):
        #     surface_wm_gt = nib.load(surf_file)
        #     vert_data_wm_gt = surface_wm_gt.darrays[0].data  # 获取顶点数据
        #     face_data_wm_gt = surface_wm_gt.darrays[1].data  # 获取面数据

        #     # 这里可以继续处理表面数据
        #     # 例如：将顶点数据转为PyTorch张量
        #     vert_tensor_wm_gt = torch.tensor(vert_data_wm_gt).float().to(device)
        #     face_tensor_wm_gt = torch.tensor(face_data_wm_gt).long().to(device)
        # else:
        #     logger.warning(f"Surface file not found for subject {subj_id}")
        
        # print('vol_data.shape:', vol_data.shape)

        # print('vol_in.shape:', vol_in.shape)

        for surf_hemi in ['left', 'right']:

            if surf_hemi == 'right': # TODO
                break

            # ============ Surface Reconstruction ============
            logger.info('----------------------------------------')
            logger.info('Surface reconstruction ({}) starts ...'.format(surf_hemi))
            t_surf_start = time.time()

            # set model, input vertices and faces
            if surf_hemi == 'left':
                nn_surf_wm = nn_surf_left_wm
                nn_surf_pial = nn_surf_left_pial
                # clip the left hemisphere
                vol_in = vol_in[:,:,96:] # TODO: adapt the ratio [256, 304, 256] [64, 112(118)]
                vert_in = vert_left_in
                face_in = face_left_in
            elif surf_hemi == 'right':
                nn_surf_wm = nn_surf_right_wm
                nn_surf_pial = nn_surf_right_pial
                # clip the right hemisphere
                vol_in = vol_in[:,:,:160] # TODO: adapt the ratio
                vert_in = vert_right_in
                face_in = face_right_in

            # wm and pial surfaces reconstruction
            with torch.no_grad():
                # 读取地面真值（GT）表面
                subj_dir = in_dir + subj_id
                surf_gt = nib.load(os.path.join(subj_dir, f"{subj_id}.L.white.native_160k.surf.gii"))
                
                # 获取地面真值顶点和面数据
                vert_gt = surf_gt.agg_data('pointset')
                face_gt = surf_gt.agg_data('triangle')[:, [2, 1, 0]]

                # 将顶点应用仿射变换
                vert_gt = apply_affine_mat(vert_gt, np.linalg.inv(affine_in)).astype(np.float32)

                # 对左半球的顶点进行平移
                if surf_hemi == 'left':
                    vert_gt[:, 0] -= 96  # 这里的 96 是根据具体的实验调整

                # 转换为 PyTorch 张量
                vert_gt_tensor = torch.tensor(vert_gt).float().to(device)
                face_gt_tensor = torch.tensor(face_gt).long().to(device)
                
                vert_wm = nn_surf_wm(vert_in, vol_in, n_steps=7) # TODO
                vert_wm = taubin_smooth(vert_wm, face_in, n_iters=5) # TODO
                vert_pial = nn_surf_pial(vert_wm, vol_in, n_steps=7) # TODO
                vert_wm_pred_tensor = torch.tensor(vert_wm[0].cpu().numpy()).float().to(device)

                pointcloud_pred = Pointclouds(points=[vert_wm_pred_tensor])  # [minibatch, num_points, 3]
                pointcloud_gt = Pointclouds(points=[vert_gt_tensor])  # [minibatch, num_points, 3]

                # 计算 Chamfer 损失
                chamfer_loss_wm = chamfer_distance(pointcloud_pred, pointcloud_gt)[0]
                chamfer_loss_wm = torch.tensor(chamfer_loss_wm) 
                chamfer_losses = list(chamfer_losses)
                chamfer_losses.append(chamfer_loss_wm.item())
                
                _, match_counts = chamfer_with_match_count(pointcloud_pred, pointcloud_gt)
                match_counts_tensor = match_counts[0]  # 取出列表里的 Tensor
                repeated_matches = (match_counts_tensor > 1).sum()  # 统计大于1的个数
                repeat_loss = repeated_matches.float().mean()   # 可作为损失函数值

                repeat_losses = list(repeat_losses)
                repeat_losses.append(repeat_loss.item())
                 


                # 打印和记录损失
                logger.info(f"Chamfer loss for wm surface: {chamfer_loss_wm.item()}")


            # torch.Tensor -> numpy.array
            vert_wm_align = vert_wm[0].cpu().numpy()
            vert_pial_align = vert_pial[0].cpu().numpy()
            face_align = face_in[0].cpu().numpy()

            # transform vertices to original space
            if surf_hemi == 'left':
                # pad the left hemisphere to full brain
                vert_wm_orig = vert_wm_align + [96,0,0] # TODO: adapt the ratio
                vert_pial_orig = vert_pial_align + [96,0,0] # TODO: adapt the ratio
            elif surf_hemi == 'right':
                vert_wm_orig = vert_wm_align.copy()
                vert_pial_orig = vert_pial_align.copy()
            
            vert_wm_orig = apply_affine_mat(
                vert_wm_orig, affine_in)
            vert_pial_orig = apply_affine_mat(
                vert_pial_orig, affine_in)
            face_orig = face_align[:,[2,1,0]]
            # midthickness surface
            vert_mid_orig = (vert_wm_orig + vert_pial_orig)/2

            # save as .surf.gii
            save_gifti_surface(
                vert_wm_orig, face_orig,
                save_dir=subj_out_dir+'_01monai_hemi-'+surf_hemi+'_wm_best.surf.gii',
                surf_hemi=surf_hemi, surf_type='wm')
            # save_gifti_surface(
            #     vert_pial_orig, face_orig, 
            #     save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_pial.surf.gii',
            #     surf_hemi=surf_hemi, surf_type='pial')
            # save_gifti_surface(
            #     vert_mid_orig, face_orig, 
            #     save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_midthickness.surf.gii',
            #     surf_hemi=surf_hemi, surf_type='midthickness')

            # send to gpu for the following processing
            vert_wm = torch.Tensor(vert_wm_orig).unsqueeze(0).to(device)
            vert_pial = torch.Tensor(vert_pial_orig).unsqueeze(0).to(device)
            vert_mid = torch.Tensor(vert_mid_orig).unsqueeze(0).to(device)
            face = torch.LongTensor(face_orig).unsqueeze(0).to(device)

            vert_gt_wm = torch.tensor(vert_wm_orig, dtype=torch.float32, device=device)  # ground truth wm
            # vert_gt_pial = torch.tensor(vert_pial_orig, dtype=torch.float32, device=device)  # ground truth pial

            t_surf_end = time.time()
            t_surf = t_surf_end - t_surf_start
            logger.info('Surface reconstruction ({}) ends. Runtime: {} sec.'.format(
                surf_hemi, np.round(t_surf, 4)))
            
            
            # ============ Cortical Feature Estimation ============
            logger.info('----------------------------------------')
            logger.info('Feature estimation ({}) starts ...'.format(surf_hemi))
            t_feature_start = time.time()

            logger.info('Estimate cortical thickness ...', end=' ')
            thickness = cortical_thickness(vert_wm, vert_pial)
            thickness = metric_dilation(
                torch.Tensor(thickness[None,:,None]).to(device),
                face, n_iters=10)
            # save_gifti_metric(
            #     metric=thickness,
            #     save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_thickness.shape.gii',
            #     surf_hemi=surf_hemi, metric_type='thickness')
            logger.info('Done.')
            
            logger.info('Estimate curvature ...', end=' ')
            curv = curvature(vert_wm, face, smooth_iters=5)
            # save_gifti_metric(
            #     metric=curv, 
            #     save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_curv.shape.gii',
            #     surf_hemi=surf_hemi, metric_type='curv')
            logger.info('Done.')

            logger.info('Estimate sulcal depth ...', end=' ')
            sulc = sulcal_depth(vert_wm, face, verbose=False)
            # save_gifti_metric(
            #     metric=sulc,
            #     save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_sulc.shape.gii',
            #     surf_hemi=surf_hemi, metric_type='sulc')
            logger.info('Done.')
            t_feature_end = time.time()
            t_feature = t_feature_end - t_feature_start
            logger.info('Cortical Feature Estimation ends. Runtime: {} sec.'.format(
                np.round(t_feature, 4)))


            # ============ Surface Inflation ============
            logger.info('----------------------------------------')
            logger.info('Surface inflation ({}) starts ...'.format(surf_hemi))
            t_inflate_start = time.time()

            # create inflated and very_inflated surfaces
            # if device is cpu, use wb_command for inflation (faster)
            if device == 'cpu':
                vert_inflated_orig, vert_vinflated_orig = \
                wb_generate_inflated_surfaces(
                    subj_out_dir, surf_hemi, iter_scale=3.0)
            else:  # cuda acceleration
                vert_inflated, vert_vinflated = generate_inflated_surfaces(
                    vert_mid, face, iter_scale=3.0)
                vert_inflated_orig = vert_inflated[0].cpu().numpy()
                vert_vinflated_orig = vert_vinflated[0].cpu().numpy()

            # save as .surf.gii
            # save_gifti_surface(
            #     vert_inflated_orig, face_orig, 
            #     save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_inflated.surf.gii',
            #     surf_hemi=surf_hemi, surf_type='inflated')
            # save_gifti_surface(
            #     vert_vinflated_orig, face_orig, 
            #     save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_vinflated.surf.gii',
            #     surf_hemi=surf_hemi, surf_type='vinflated')

            t_inflate_end = time.time()
            t_inflate = t_inflate_end - t_inflate_start
            logger.info('Surface inflation ({}) ends. Runtime: {} sec.'.format(
                surf_hemi, np.round(t_inflate, 4)))


            
        
        logger.info('----------------------------------------')
        # clean temp data
        # os.remove(subj_out_dir+'_rigid_0GenericAffine.mat')
        # os.remove(subj_out_dir+'_affine_0GenericAffine.mat')
        # os.remove(subj_out_dir+'_ribbon.nii.gz')
        if os.path.exists(subj_out_dir+'_T1wDividedByT2w.nii.gz'):
            os.remove(subj_out_dir+'_T1wDividedByT2w.nii.gz')
        # create .spec file for visualization
        # create_wb_spec(subj_out_dir) # TODO
        t_end = time.time()
        logger.info('Finished. Total runtime: {} sec.'.format(
            np.round(t_end-t_start, 4)))
        logger.info('========================================')
        end_time = time.time()
        print(f'Total Time: {end_time - start_time}s')
        if num_subjects > 0:
            print('========================================')
            print(f'Average surface reconstruction time per subject: {total_surf_time / num_subjects:.4f} sec')
            print(f'Average cortical feature estimation time per subject: {total_feature_time / num_subjects:.4f} sec')
            print(f'Average surface inflation time per subject: {total_inflate_time / num_subjects:.4f} sec')
            print('========================================')

        chamfer_losses = np.array(chamfer_losses)   # 转换为 NumPy 数组进行计算
        max_loss = np.max(chamfer_losses)
        min_loss = np.min(chamfer_losses)
        mean_loss = np.mean(chamfer_losses)
        std_loss = np.std(chamfer_losses)

        # 打印统计值
        logger.info(f"Chamfer Loss - Max: {max_loss}, Min: {min_loss}, Mean: {mean_loss}, Std: {std_loss}")
        repeat_losses = np.array(repeat_losses)   # 转换为 NumPy 数组进行计算
        rmax_loss = np.max(repeat_losses)
        rmin_loss = np.min(repeat_losses)
        rmean_loss = np.mean(repeat_losses)
        rstd_loss = np.std(repeat_losses)
        logger.info(f"repeat Loss - Max: {rmax_loss}, Min: {rmin_loss}, Mean: {rmean_loss}, Std: {rstd_loss}")

