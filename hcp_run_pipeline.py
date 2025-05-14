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

# from preprocess import extract_image as ei
# from preprocess import get_metadata as gm # type: ignore

# ============ load hyperparameters ============
parser = argparse.ArgumentParser(description="dHCP DL Neonatal Pipeline")
# parser.add_argument('--in_dir', default='/root/autodl-tmp/datasets/', type=str,
#                     help='Diectory containing input images.')
# parser.add_argument('--out_dir', default='/root/autodl-tmp/output/test/', type=str,
#                     help='Directory for saving the output of the pipeline.')
parser.add_argument('--in_dir', default='/root/autodl-tmp/hcp_gcn/dataset_hcp_test/', type=str, # TODO
                    help='Diectory containing input images.')
parser.add_argument('--out_dir', default='/root/autodl-tmp/hcp_gcn/results_by_my_model/', type=str, # TODO
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
nn_seg_ribbon = UNet(
    C_in=1, C_hid=[16,32,64,128,128], C_out=1).to(device)

nn_seg_brain.load_state_dict( # done
    torch.load('/root/autodl-tmp/hcp_gcn/model_trained_by_hcp_10/model_seg_brain_all.pt', map_location=device))
nn_seg_ribbon.load_state_dict( # ignore
    torch.load('./seg/model/model_seg_ribbon.pt', map_location=device))

# surface reconstruction
nn_surf_left_wm = SurfDeform( # done
    C_hid=[8,16,32,64,128,128], C_in=2, inshape=[160,304,256], sigma=1.0, device=device) # TODO
nn_surf_right_wm = SurfDeform( # done
    C_hid=[8,16,32,64,128,128], C_in=1, inshape=[160,304,256], sigma=1.0, device=device)
nn_surf_left_pial = SurfDeform( # done
    C_hid=[8,16,32,32,32,32], C_in=1, inshape=[160,304,256], sigma=1.0, device=device)
nn_surf_right_pial = SurfDeform( # done
    C_hid=[8,16,32,32,32,32], C_in=1, inshape=[160,304,256], sigma=1.0, device=device)

nn_surf_left_wm.load_state_dict( # training
    torch.load('/root/autodl-tmp/hcp_gcn/model_trained_by_hcp_all/model_hemi-left_wm_all_multi_80epochs.pt', map_location=device))
nn_surf_right_wm.load_state_dict(
    torch.load('./surface/model/model_hemi-right_wm.pt', map_location=device))
nn_surf_left_pial.load_state_dict(
    torch.load('./surface/model/model_hemi-left_pial.pt', map_location=device))
nn_surf_right_pial.load_state_dict(
    torch.load('./surface/model/model_hemi-right_pial.pt', map_location=device))

# spherical projection
if sphere_type == 'fs':
    C_in_sphere = 18
elif sphere_type == 'mds':
    C_in_sphere = 6
nn_sphere_left = SphereDeform( # ignore
    C_in=C_in_sphere, C_hid=[32, 64, 128, 128, 128], device=device)
nn_sphere_right = SphereDeform( # ignore
    C_in=C_in_sphere, C_hid=[32, 64, 128, 128, 128], device=device)

nn_sphere_left.load_state_dict(
    torch.load('./sphere/model/model_hemi-left_sphere_'+sphere_type+'.pt', map_location=device))
nn_sphere_right.load_state_dict(
    torch.load('./sphere/model/model_hemi-right_sphere_'+sphere_type+'.pt', map_location=device))


# TODO: image template for affine register
# ============ load image atlas ============
# img_t1_atlas_ants = ants.image_read( 
#     './template/T1w.nii.gz')
# # both ants->nibabel and nibabel->ants need to reload the nifiti file
# # so here simply load the image again
# affine_t1_atlas = nib.load( # type: ignore  same as above
#     './template/T1w.nii.gz').affine # type: ignore

img_t1_atlas_ants = ants.image_read('/root/autodl-tmp/hcp_gcn/template/MNI152_T1_0.7mm_brain_sampled.nii.gz')

# # 降采样模板图像到目标shape（假设目标shape为 [256, 304, 256]） # TODO
# target_shape = [256, 304, 256]  # 根据你的输入shape修改
# img_t1_atlas_ants = ants.resample_image(
#     img_t1_atlas_ants, 
#     target_shape, 
#     use_voxels=True, 
#     interp_type=1  # 1表示线性插值
# )

# 重新加载模板的affine矩阵
affine_t1_atlas = nib.load('/root/autodl-tmp/hcp_gcn/template/MNI152_T1_0.7mm_brain_sampled.nii.gz').affine  # type: ignore

# ============ load input surface ============
surf_left_in = nib.load( # type: ignore
    '/root/autodl-tmp/hcp_gcn/template/lh.white.surf.gii')
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
    '/root/autodl-tmp/hcp_gcn/template/rh.white.surf.gii')
# surf_right_in = nib.load( # type: ignore
#     '/root/dhcp/hcp_template/S1200.R.inflated_MSMAll.32k_fs_LR_150k.surf.gii')
vert_right_in = surf_right_in.agg_data('pointset') # type: ignore
face_right_in = surf_right_in.agg_data('triangle') # type: ignore
vert_right_in = apply_affine_mat(
    vert_right_in, np.linalg.inv(affine_t1_atlas))
face_right_in = face_right_in[:,[2,1,0]]
vert_right_in = torch.Tensor(vert_right_in[None]).to(device)
face_right_in = torch.LongTensor(face_right_in[None]).to(device)


# ============ load input sphere ============
sphere_left_in = nib.load( # type: ignore
    './template/dhcp_week-40_hemi-left_sphere_'+sphere_type+'.surf.gii')
vert_sphere_left_in = sphere_left_in.agg_data('pointset') # type: ignore
vert_sphere_left_in = torch.Tensor(vert_sphere_left_in[None]).to(device)

sphere_right_in = nib.load( # type: ignore
    './template/dhcp_week-40_hemi-right_sphere_'+sphere_type+'.surf.gii')
vert_sphere_right_in = sphere_right_in.agg_data('pointset') # type: ignore
vert_sphere_right_in = torch.Tensor(vert_sphere_right_in[None]).to(device)


# ============ load template sphere (160k) ============
sphere_160k = nib.load('./template/sphere_163842.surf.gii') # type: ignore
vert_sphere_160k = sphere_160k.agg_data('pointset') # type: ignore
face_160k = sphere_160k.agg_data('triangle') # type: ignore
vert_sphere_160k = torch.Tensor(vert_sphere_160k[None]).to(device)
face_160k = torch.LongTensor(face_160k[None]).to(device)
neigh_order_160k = get_neighs_order()[0]  # neighbors


# ============ load pre-computed barycentric coordinates ============
# for sphere interpolation
barycentric_left = nib.load( # type: ignore
    './template/dhcp_week-40_hemi-left_barycentric_'+sphere_type+'.gii')
bc_coord_left = barycentric_left.agg_data('pointset') # type: ignore
face_left_id = barycentric_left.agg_data('triangle') # type: ignore

barycentric_right = nib.load( # type: ignore
    './template/dhcp_week-40_hemi-right_barycentric_'+sphere_type+'.gii')
bc_coord_right = barycentric_right.agg_data('pointset') # type: ignore
face_right_id = barycentric_right.agg_data('triangle') # type: ignore

# python /home/sywang/dhcp/run_pipeline.py --in_dir=pipeline_test/ --out_dir=output/ --T2=_T2w.nii.gz --T1=_T1w.nii.gz --sphere_proj=fs --device=cuda:2 
# ============ dHCP DL-based neonatal pipeline ============
if __name__ == '__main__':
    start_time = time.time()
    
    subj_list = sorted(glob.glob(in_dir + '**/*' + t2_suffix, recursive=True))

    for subj_t2_dir in tqdm(subj_list): # switch to t2
            

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
        
        
        # ============ Load Data ============
        # logger.info('Loading T2w_proc_down image ...', end=' ')
        # img_nib = nib.load(subj_t2_dir)
        # img_data = img_nib.get_fdata()
        # affine_t2_align = img_nib.affine
        # logger.info('Done.')
        
    
        # vol_t2_align= torch.tensor(img_data[None, None]).float().to(device)
        # vol_t2_align /=vol_t2_align.max()  # normalize to [0,1]

        # print("vol_t2_align shape:", vol_t2_align.shape)

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
        
        print('vol_data.shape:', vol_data.shape)

        print('vol_in.shape:', vol_in.shape)





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
                vert_wm = nn_surf_wm(vert_in, vol_in, n_steps=7) # TODO
                vert_wm = taubin_smooth(vert_wm, face_in, n_iters=5) # TODO
                # vert_pial = nn_surf_pial(vert_wm, vol_in, n_steps=7) # TODO

            # torch.Tensor -> numpy.array
            vert_wm_align = vert_wm[0].cpu().numpy()
            # vert_pial_align = vert_pial[0].cpu().numpy()
            face_align = face_in[0].cpu().numpy()

            # transform vertices to original space
            if surf_hemi == 'left':
                # pad the left hemisphere to full brain
                vert_wm_orig = vert_wm_align + [96,0,0] # TODO: adapt the ratio
                # vert_pial_orig = vert_pial_align + [96,0,0] # TODO: adapt the ratio
            elif surf_hemi == 'right':
                vert_wm_orig = vert_wm_align.copy()
                # vert_pial_orig = vert_pial_align.copy()
            vert_wm_orig = apply_affine_mat(
                vert_wm_orig, affine_in)
            # vert_pial_orig = apply_affine_mat(
            #     vert_pial_orig, affine_t1_align)
            face_orig = face_align[:,[2,1,0]]
            # midthickness surface
            # vert_mid_orig = (vert_wm_orig + vert_pial_orig)/2

            # save as .surf.gii
            save_gifti_surface(
                vert_wm_orig, face_orig,
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_wm.surf.gii',
                surf_hemi=surf_hemi, surf_type='wm')
            # # save_gifti_surface(
            # #     vert_pial_orig, face_orig, 
            # #     save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_pial.surf.gii',
            # #     surf_hemi=surf_hemi, surf_type='pial')
            # # save_gifti_surface(
            #     vert_mid_orig, face_orig, 
            #     save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_midthickness.surf.gii',
            #     surf_hemi=surf_hemi, surf_type='midthickness')

            # send to gpu for the following processing
            vert_wm = torch.Tensor(vert_wm_orig).unsqueeze(0).to(device)
            # vert_pial = torch.Tensor(vert_pial_orig).unsqueeze(0).to(device)
            # vert_mid = torch.Tensor(vert_mid_orig).unsqueeze(0).to(device)
            face = torch.LongTensor(face_orig).unsqueeze(0).to(device)

            t_surf_end = time.time()
            t_surf = t_surf_end - t_surf_start
            logger.info('Surface reconstruction ({}) ends. Runtime: {} sec.'.format(
                surf_hemi, np.round(t_surf, 4)))
            
            '''
            # ============ Cortical Feature Estimation ============
            logger.info('----------------------------------------')
            logger.info('Feature estimation ({}) starts ...'.format(surf_hemi))
            t_feature_start = time.time()

            logger.info('Estimate cortical thickness ...', end=' ')
            thickness = cortical_thickness(vert_wm, vert_pial)
            thickness = metric_dilation(
                torch.Tensor(thickness[None,:,None]).to(device),
                face, n_iters=10)
            save_gifti_metric(
                metric=thickness,
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_thickness.shape.gii',
                surf_hemi=surf_hemi, metric_type='thickness')
            logger.info('Done.')
            
            logger.info('Estimate curvature ...', end=' ')
            curv = curvature(vert_wm, face, smooth_iters=5)
            save_gifti_metric(
                metric=curv, 
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_curv.shape.gii',
                surf_hemi=surf_hemi, metric_type='curv')
            logger.info('Done.')

            logger.info('Estimate sulcal depth ...', end=' ')
            sulc = sulcal_depth(vert_wm, face, verbose=False)
            save_gifti_metric(
                metric=sulc,
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_sulc.shape.gii',
                surf_hemi=surf_hemi, metric_type='sulc')
            logger.info('Done.')

            continue # TODO

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
            save_gifti_surface(
                vert_inflated_orig, face_orig, 
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_inflated.surf.gii',
                surf_hemi=surf_hemi, surf_type='inflated')
            save_gifti_surface(
                vert_vinflated_orig, face_orig, 
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_vinflated.surf.gii',
                surf_hemi=surf_hemi, surf_type='vinflated')

            t_inflate_end = time.time()
            t_inflate = t_inflate_end - t_inflate_start
            logger.info('Surface inflation ({}) ends. Runtime: {} sec.'.format(
                surf_hemi, np.round(t_inflate, 4)))


            # ============ Spherical Mapping ============
            logger.info('----------------------------------------')
            logger.info('Spherical mapping ({}) starts ...'.format(surf_hemi))
            t_sphere_start = time.time()

            # set model, input vertices and faces
            if surf_hemi == 'left':
                nn_sphere = nn_sphere_left
                vert_sphere_in = vert_sphere_left_in
                bc_coord = bc_coord_left
                face_id = face_left_id
            elif surf_hemi == 'right':
                nn_sphere = nn_sphere_right
                vert_sphere_in = vert_sphere_right_in
                bc_coord = bc_coord_right
                face_id = face_right_id

            # interpolate to 160k template
            vert_wm_160k = (vert_wm_orig[face_id] * bc_coord[...,None]).sum(-2)
            vert_wm_160k = torch.Tensor(vert_wm_160k[None]).to(device)
            if sphere_type == 'fs':
                # input metric features
                neigh_wm_160k = vert_wm_160k[:, neigh_order_160k].reshape(
                    vert_wm_160k.shape[0], vert_wm_160k.shape[1], 7, 3)[:,:,:-1]
                edge_wm_160k = (neigh_wm_160k - vert_wm_160k[:,:,None]).norm(dim=-1)
                area_wm_160k = 0.5*torch.norm(torch.cross(
                    neigh_wm_160k[:,:,[0,1,2,3,4,5]] - vert_wm_160k[:,:,None],
                    neigh_wm_160k[:,:,[1,2,3,4,5,0]] - vert_wm_160k[:,:,None]), dim=-1)
                # final input features
                feat_160k = torch.cat(
                    [vert_sphere_160k, vert_wm_160k, edge_wm_160k, area_wm_160k], dim=-1)
            elif sphere_type == 'mds':
                feat_160k = torch.cat(
                    [vert_sphere_160k, vert_wm_160k], dim=-1)

            with torch.no_grad():
                vert_sphere = nn_sphere(
                    feat_160k, vert_sphere_in, n_steps=7)
                
            # compute metric distortion
            edge = torch.cat([
                face[0,:,[0,1]],
                face[0,:,[1,2]],
                face[0,:,[2,0]]], dim=0).T
            # edge_distort = edge_distortion(vert_sphere, vert_wm, edge).item()
            # area_distort = area_distortion(vert_sphere, vert_wm, face).item()
            # logger.info('Edge distortion: {}mm'.format(np.round(edge_distort, 4)))
            # logger.info('Area distortion: {}mm^2'.format(np.round(area_distort, 4)))
            
            # save as .surf.gii
            vert_sphere = vert_sphere[0].cpu().numpy()
            save_gifti_surface(
                vert_sphere, face_orig, 
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_sphere.surf.gii',
                surf_hemi=surf_hemi, surf_type='sphere')

            t_sphere_end = time.time()
            t_sphere = t_sphere_end - t_sphere_start
            logger.info('Spherical mapping ({}) ends. Runtime: {} sec.'.format(
                surf_hemi, np.round(t_sphere, 4)))


            # # ============ Cortical Feature Estimation ============
            # logger.info('----------------------------------------')
            # logger.info('Feature estimation ({}) starts ...'.format(surf_hemi))
            # t_feature_start = time.time()

            # logger.info('Estimate cortical thickness ...', end=' ')
            # thickness = cortical_thickness(vert_wm, vert_pial)
            # thickness = metric_dilation(
            #     torch.Tensor(thickness[None,:,None]).to(device),
            #     face, n_iters=10)
            # save_gifti_metric(
            #     metric=thickness,
            #     save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_thickness.shape.gii',
            #     surf_hemi=surf_hemi, metric_type='thickness')
            # logger.info('Done.')

            # logger.info('Estimate curvature ...', end=' ')
            # curv = curvature(vert_wm, face, smooth_iters=5)
            # save_gifti_metric(
            #     metric=curv, 
            #     save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_curv.shape.gii',
            #     surf_hemi=surf_hemi, metric_type='curv')
            # logger.info('Done.')

            # logger.info('Estimate sulcal depth ...', end=' ')
            # sulc = sulcal_depth(vert_wm, face, verbose=False)
            # save_gifti_metric(
            #     metric=sulc,
            #     save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_sulc.shape.gii',
            #     surf_hemi=surf_hemi, metric_type='sulc')
            # logger.info('Done.')

            
            # ============ myelin map estimation ============
            # estimate myelin map based on
            # t1-to-t2 ratio, midthickness surface, 
            # cortical thickness and cortical ribbon

            if t2_exists:
                logger.info('Estimate myelin map ...', end=' ')
                myelin = myelin_map(
                    subj_dir=subj_out_dir, surf_hemi=surf_hemi)
                # metric dilation
                myelin = metric_dilation(
                    torch.Tensor(myelin[None,:,None]).to(device),
                    face, n_iters=10)
                # save myelin map
                save_gifti_metric(
                    metric=myelin,
                    save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_myelinmap.shape.gii',
                    surf_hemi=surf_hemi, metric_type='myelinmap')
                
                # smooth myelin map
                smoothed_myelin = smooth_myelin_map(
                    subj_dir=subj_out_dir, surf_hemi=surf_hemi)
                save_gifti_metric(
                    metric=smoothed_myelin, 
                    save_dir=subj_out_dir+'_hemi-'+surf_hemi+\
                             '_smoothed_myelinmap.shape.gii',
                    surf_hemi=surf_hemi,
                    metric_type='smoothed_myelinmap')
                logger.info('Done.')

            t_feature_end = time.time()
            t_feature = t_feature_end - t_feature_start
            logger.info('Feature estimation ({}) ends. Runtime: {} sec.'.format(
                surf_hemi, np.round(t_feature, 4)))
            '''
        
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
