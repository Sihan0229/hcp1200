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
    area_distortion,
    # combine_edge_distortion,
    # combine_area_distortion,
    # hausdorff_distance,
    # chamfer_distance
    )

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



# # ============ load hyperparameters ============
# parser = argparse.ArgumentParser(description="dHCP DL surface restruction training Pipeline")
# parser.add_argument('--in_dir', default='./in_dir/', type=str,
#                     help='Diectory containing input images.')
# parser.add_argument('--out_dir', default='./out_dir/', type=str,
#                     help='Directory for saving the output of the pipeline.')
# parser.add_argument('--restore_suffix', default='_desc_T2w_restore_brain.nii.gz', type=str,
#                     help='Suffix of T2 restore image file.')
# parser.add_argument('--hemi', default='left', type=str,
#                     help='Training for left or right part of brain')
# parser.add_argument('--surface', default='wm', type=str,
#                     help='Training for wm or pial of brain')                   
# parser.add_argument('--device', default='cuda', type=str,
#                     help='Device for running the pipeline: [cuda, cpu]')
# parser.add_argument('--verbose', action='store_true',
#                     help='Print debugging information.')
# args = parser.parse_args()

# in_dir = args.in_dir
# out_dir = args.out_dir
# restore_suffix = args.restore_suffix
hemi = 'left'
surface = 'wm'

device = 'cuda'
verbose = 'store_true'

max_regist_iter = 5
min_regist_dice = 0.9


# ============ load nn model ============

# surface reconstruction
nn_surf_left_wm = SurfDeform(
    C_hid=[8,16,32,64,128,128], C_in=1, sigma=1.0, device=device)
nn_surf_right_wm = SurfDeform(
    C_hid=[8,16,32,64,128,128], C_in=1, sigma=1.0, device=device)
nn_surf_left_pial = SurfDeform(
    C_hid=[8,16,32,32,32,32], C_in=1, sigma=1.0, device=device)
nn_surf_right_pial = SurfDeform(
    C_hid=[8,16,32,32,32,32], C_in=1, sigma=1.0, device=device)

nn_surf_left_wm.load_state_dict(
    torch.load('./surface/model/model_hemi-left_wm.pt', map_location=device))
nn_surf_right_wm.load_state_dict(
    torch.load('./surface/model/model_hemi-right_wm.pt', map_location=device))
nn_surf_left_pial.load_state_dict(
    torch.load('./surface/model/model_hemi-left_pial.pt', map_location=device))
nn_surf_right_pial.load_state_dict(
    torch.load('./surface/model/model_hemi-right_pial.pt', map_location=device))


# ============ load image atlas ============
# 皮层重建要调用
img_t2_atlas_ants = ants.image_read(
    './template/dhcp_week-40_template_T2w.nii.gz')
# both ants->nibabel and nibabel->ants need to reload the nifiti file
# so here simply load the image again
affine_t2_atlas = nib.load(
    './template/dhcp_week-40_template_T2w.nii.gz').affine


# ============ load input surface ============
# 皮层重建部分直接调用这些原始表面，有用
surf_left_in = nib.load(
    './template/dhcp_week-40_hemi-left_init.surf.gii')
vert_left_in = surf_left_in.agg_data('pointset')
face_left_in = surf_left_in.agg_data('triangle')
vert_left_in = apply_affine_mat(
    vert_left_in, np.linalg.inv(affine_t2_atlas))
vert_left_in = vert_left_in - [64,0,0]
face_left_in = face_left_in[:,[2,1,0]]
vert_left_in = torch.Tensor(vert_left_in[None]).to(device)
face_left_in = torch.LongTensor(face_left_in[None]).to(device)

surf_right_in = nib.load(
    './template/dhcp_week-40_hemi-right_init.surf.gii')
vert_right_in = surf_right_in.agg_data('pointset')
face_right_in = surf_right_in.agg_data('triangle')
vert_right_in = apply_affine_mat(
    vert_right_in, np.linalg.inv(affine_t2_atlas))
face_right_in = face_right_in[:,[2,1,0]]
vert_right_in = torch.Tensor(vert_right_in[None]).to(device)
face_right_in = torch.LongTensor(face_right_in[None]).to(device)




# ============ dHCP DL-based surface reconstruction pipeline ============
# if __name__ == '__main__':
#     subj_list = sorted(glob.glob(os.path.join(in_dir, '**', '*' + restore_suffix), recursive=True))
#     print('subj_list:',subj_list)
#     for subj_t2_dir in tqdm(subj_list):
#         # extract subject id
#         subj_id = subj_t2_dir.split('/')[-1][:-len(restore_suffix)]
#         print('subj_id:',subj_id)
#         # directory for saving output: out_dir/subj_id/
#         subj_out_dir = out_dir + subj_id + '/'
#         print('subj_out_dir:',subj_out_dir)

#         # create output directory
#         if not os.path.exists(subj_out_dir):
#             os.mkdir(subj_out_dir)
#             # add subject id as prefix
#         subj_out_dir = subj_out_dir + subj_id

#         t2_restore_brain_path = os.path.join(in_dir, subj_id, subj_id + restore_suffix)
#         print('t2_restore_brain_path:',t2_restore_brain_path)

#         # initialize logger
#         logger = Logging(subj_out_dir)
#         # start processing
#         logger.info('========================================')
#         logger.info('Start processing subject: {}'.format(subj_id))
#         t_start = time.time()

    
#         # 使用 ANTs 读入处理后的图像
#         img_t2_proc_ants = ants.image_read(t2_restore_brain_path)

#         # 使用 nibabel 获取 affine 矩阵
#         affine_t2_proc = nib.load(t2_restore_brain_path).affine

#         # ============ Affine Registration ============
#         logger.info('----------------------------------------')
#         logger.info('Affine registration starts ...')

#         # ants affine registration
#         img_t2_align_ants, affine_t2_align, _, _, align_dice =\
#         registration(
#             img_move_ants=img_t2_proc_ants, # 这一个是计算结果，需要导入
#             img_fix_ants=img_t2_atlas_ants, # 模板
#             affine_fix=affine_t2_atlas, # 模板
#             out_prefix=subj_out_dir, 
#             max_iter=max_regist_iter, # args
#             min_dice=min_regist_dice, # args
#             verbose=verbose) # args
        
#         args_t2_align_ants = (
#             img_t2_align_ants.origin,
#             img_t2_align_ants.spacing,
#             img_t2_align_ants.direction)
#         img_t2_align = img_t2_align_ants.numpy()

#         vol_t2_align = torch.Tensor(img_t2_align[None,None]).to(device)
#         vol_t2_align = (vol_t2_align / vol_t2_align.max()).float()
        
#         #print('vol_t2_align:', vol_t2_align.shape)
  
#         # ============ Surface Reconstruction ============
#         logger.info('----------------------------------------')
#         logger.info('Surface reconstruction ({}) starts ...'.format(surf_hemi))

        # set model, input vertices and faces
if hemi == 'left':
    nn_surf_wm = nn_surf_left_wm
    nn_surf_pial = nn_surf_left_pial
    # clip the left hemisphere
    # vol_in = vol_t2_align[:,:,64:]
    vert_in = vert_left_in
    face_in = face_left_in
elif hemi == 'right':
    nn_surf_wm = nn_surf_right_wm
    nn_surf_pial = nn_surf_right_pial
    # clip the right hemisphere
    # vol_in = vol_t2_align[:,:,:112]
    vert_in = vert_right_in
    face_in = face_right_in
vol_in_path = '/root/autodl-tmp/hcp_gcn/datasets_for_surface_resconstruction_testing/sub-CC00065XX08_ses-18600/sub-CC00065XX08_ses-18600_hemi-left_vol_in.pt'
vol_in = torch.load(vol_in_path)
# wm and pial surfaces reconstruction
with torch.no_grad():
    
    
    vert_wm = nn_surf_wm(vert_in, vol_in, n_steps=7)
    vert_wm = taubin_smooth(vert_wm, face_in, n_iters=5)
    
    # vert_pial = nn_surf_pial(vert_wm, vol_in, n_steps=7)

# torch.Tensor -> numpy.array
vert_wm_align = vert_wm[0].cpu().numpy()
# vert_pial_align = vert_pial[0].cpu().numpy()
face_align = face_in[0].cpu().numpy()

# transform vertices to original space
if hemi == 'left':
    # pad the left hemisphere to full brain
    vert_wm_orig = vert_wm_align + [64,0,0]
    # vert_pial_orig = vert_pial_align + [64,0,0]
elif hemi == 'right':
    vert_wm_orig = vert_wm_align.copy()
    # vert_pial_orig = vert_pial_align.copy()
vert_wm_orig = apply_affine_mat(
    vert_wm_orig, affine_t2_align)
# vert_pial_orig = apply_affine_mat(
#     vert_pial_orig, affine_t2_align)
face_orig = face_align[:,[2,1,0]]
    
# midthickness surface
# vert_mid_orig = (vert_wm_orig + vert_pial_orig)/2

# save as .surf.gii
save_gifti_surface(
    vert_wm_orig, face_orig,
    save_dir='/root/autodl-tmp/hcp_gcn/datasets_for_surface_resconstruction_testing/sub-CC00065XX08_ses-18600/sub-CC00065XX08_ses-18600_hemi-'+surf_hemi+'_wm_simple.surf.gii',
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
# vert_pial = torch.Tensor(vert_pial_orig).unsqueeze(0).to(device)
# vert_mid = torch.Tensor(vert_mid_orig).unsqueeze(0).to(device)
face = torch.LongTensor(face_orig).unsqueeze(0).to(device)

    
logger.info('----------------------------------------')
t_end = time.time()
logger.info('Finished. Total runtime: {} sec.'.format(
    np.round(t_end-t_start, 4)))
logger.info('========================================')
