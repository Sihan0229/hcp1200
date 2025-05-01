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



# ============ load hyperparameters ============
parser = argparse.ArgumentParser(description="dHCP DL Neonatal training Pipeline")
parser.add_argument('--in_dir', default='./in_dir/', type=str,
                    help='Diectory containing input images.')
parser.add_argument('--out_dir', default='./out_dir/', type=str,
                    help='Directory for saving the output of the pipeline.')
parser.add_argument('--T2', default='_T2w.nii.gz', type=str,
                    help='Suffix of T2 image file.')
parser.add_argument('--T1', default='_T1w.nii.gz', type=str,
                    help='Suffix of T1 image file.')
parser.add_argument('--sphere_proj', default='fs', type=str,
                    help='The method of spherical projection: [fs, mds].')
parser.add_argument('--device', default='cuda', type=str,
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

nn_seg_brain.load_state_dict(
    torch.load('./seg/model/model_seg_brain.pt', map_location=device))
nn_seg_ribbon.load_state_dict(
    torch.load('./seg/model/model_seg_ribbon.pt', map_location=device))

# ============ load image atlas ============
img_t1_t2_ratio = ants.image_read(
    './template/dhcp_week-40_template_T2w.nii.gz')
# both ants->nibabel and nibabel->ants need to reload the nifiti file
# so here simply load the image again
img_t1_t2_ratio = nib.load(
    './template/dhcp_week-40_template_T2w.nii.gz').affine



# ============ dHCP DL-based neonatal pipeline ============
if __name__ == '__main__':
    subj_list = sorted(glob.glob(in_dir+'*'+t2_suffix))
    for subj_t2_dir in tqdm(subj_list):
        # extract subject id
        subj_id = subj_t2_dir.split('/')[-1][:-len(t2_suffix)]
        # check if T1 image exists
        if t1_suffix:
            subj_t1_dir = in_dir + subj_id + t1_suffix
        t1_exists = False
        if os.path.exists(subj_t1_dir):
            t1_exists = True

        # directory for saving output: out_dir/subj_id/
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
        logger.info('Load T2 image ...', end=' ')
        # load original T2 image
        img_t2_orig_ants = ants.image_read(subj_t2_dir)
        img_t2_orig = img_t2_orig_ants.numpy()

        # ants image produces inaccurate affine matrix
        # reload the nifti file to get the affine matrix
        img_t2_orig_nib = nib.load(subj_t2_dir)
        affine_t2_orig = img_t2_orig_nib.affine # 有用

        # args for converting numpy.array to ants image
        args_t2_orig_ants = (
            img_t2_orig_ants.origin,
            img_t2_orig_ants.spacing,
            img_t2_orig_ants.direction)
        logger.info('Done.')
        
        
        # load t1 image if exists
        if t1_exists:
            logger.info('Load T1 image ...', end=' ')
            # load original T1 image
            img_t1_orig_ants = ants.image_read(subj_t1_dir)
            img_t1_orig = img_t1_orig_ants.numpy()
            logger.info('Done.')
        

        
        # ============ brain extraction ============
        logger.info('----------------------------------------')
        logger.info('Brain extraction starts ...')
        t_brain_start = time.time()
        
        vol_t2_orig = torch.Tensor(img_t2_orig[None,None]).to(device)
        # 体数据缩小 三线性插值
        vol_t2_orig_down = F.interpolate(
            vol_t2_orig, size=[160,208,208], mode='trilinear')
        # 归一化 MRI 数据
        vol_in = (vol_t2_orig_down / vol_t2_orig_down.max()).float()
        # 用深度学习模型预测脑掩码
        with torch.no_grad():
            brain_mask_pred = torch.sigmoid(nn_seg_brain(vol_in))
            brain_mask_pred = F.interpolate(
                brain_mask_pred, size=vol_t2_orig.shape[2:], mode='trilinear')
        # threshold to binary mask 转换为二值脑掩码
        brain_mask_orig = (brain_mask_pred[0,0]>0.5).float().cpu().numpy()
        
        # save brain mask
        save_numpy_to_nifti(
            brain_mask_orig, affine_t2_orig,
            subj_out_dir+'_brain_mask.nii.gz')
        
        t_brain_end = time.time()
        t_brain = t_brain_end - t_brain_start
        logger.info('Brain extraction ends. Runtime: {} sec.'.format(
            np.round(t_brain,4)))
        
        
        # ============ N4 bias field correction ============
        logger.info('----------------------------------------')
        logger.info('Bias field correction starts ...')
        t_bias_start = time.time()

        # N4 bias field correction
        brain_mask_ants = ants.from_numpy(
            brain_mask_orig, *args_t2_orig_ants)
        
        img_t2_restore_ants = n4_bias_field_correction(
            img_t2_orig_ants, brain_mask_ants,
            shrink_factor=4,
            convergence={"iters": [50, 50, 50], "tol": 0.001},
            spline_param=100,
            verbose=verbose)
        
        img_t2_restore = img_t2_restore_ants.numpy()
        
        # brain extracted and bias corrected image
        img_t2_proc = img_t2_restore * brain_mask_orig # 有用
        img_t2_proc_ants = ants.from_numpy(
            img_t2_proc, *args_t2_orig_ants)

        # save brain extracted and bias corrected T2w image
        save_numpy_to_nifti(
            img_t2_proc, affine_t2_orig,
            subj_out_dir+'_T2w_restore_brain.nii.gz')
        
        t_bias_end = time.time()
        t_bias = t_bias_end - t_bias_start
        logger.info('T2 bias field correction ends. Runtime: {} sec.'.format(
            np.round(t_bias,4)))
        
        
        
        # ============ Cortical Ribbon Segmentation ============
        logger.info('----------------------------------------')
        logger.info('Cortical ribbon segmentation starts ...')
        t_ribbon_start = time.time()
        
        # 此处是N4处理过的img_t2_proc = img_t2_restore * brain_mask_orig
        vol_t2_proc = torch.Tensor(img_t2_proc[None,None]).to(device)
        vol_t2_proc_down = F.interpolate(
            vol_t2_proc, size=[160,208,208], mode='trilinear')
        vol_in = (vol_t2_proc_down / vol_t2_proc_down.max()).float()
        with torch.no_grad():
            ribbon_pred = torch.sigmoid(nn_seg_ribbon(vol_in))
            ribbon_pred = F.interpolate(
                ribbon_pred, size=vol_t2_proc.shape[2:], mode='trilinear')
        # threshold to binary mask
        ribbon_orig = (ribbon_pred[0,0]>0.5).float().cpu().numpy()

        # save ribbon file
        save_numpy_to_nifti(
            ribbon_orig, affine_t2_orig, subj_out_dir+'_ribbon.nii.gz')

        t_ribbon_end = time.time()
        t_ribbon = t_ribbon_end - t_ribbon_start
        logger.info('Cortical ribbon segmentation ends. Runtime: {} sec.'.format(
            np.round(t_ribbon, 4)))
        

        logger.info('----------------------------------------')
        
        t_end = time.time()
        logger.info('Finished. Total runtime: {} sec.'.format(
            np.round(t_end-t_start, 4)))
        logger.info('========================================')
