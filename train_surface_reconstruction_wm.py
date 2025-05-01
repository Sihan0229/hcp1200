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
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import ants
from ants.utils.bias_correction import n4_bias_field_correction

from seg.unet import UNet
from surface.net import SurfDeform
from sphere.net.sunet import SphereDeform
from surface.loss import (
    edge_distortion,
    area_distortion,
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




# =============数据集导入==============
# 主数据导入函数
def load_dataset(in_dir, surf_hemi='left'):
    subj_paths = [os.path.join(in_dir, subj) for subj in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, subj))]
    print('subj_paths:', subj_paths)
    dataset = []

    for path in tqdm(subj_paths, desc='Loading data'):
        subj_id = os.path.basename(path)
        print('subj_id:', subj_id)

        vol_in_path = os.path.join(path, f"{subj_id}_hemi-{hemi}_vol_in.pt")
        vert_wm_path = os.path.join(path, f"{subj_id}_hemi-{hemi}_vert_wm_before_smooth.pt")
        # face_wm_path = os.path.join(path, f"{subj_id}_hemi-{hemi}_vol_in.pt")
  
        # 读取vol_in
        vol_in = torch.load(vol_in_path)
        vert_wm = torch.load(vert_wm_path)
        # face_wm = torch.load(face_wm_path)

        dataset.append({
            'subj_id': subj_id,
            'vol_in': vol_in,
            'vert_wm': vert_wm,
            # 'face_wm': face_wm
        })

    return dataset



# =============模型数据集类定义==============
class SurfaceDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'subj_id': item['subj_id'],
            'vol': item['vol_in'].float(),
            'vert_wm': item['vert_wm'].float(),
            # 'face_wm': item['face_wm'].long()
        }

# =============损失函数==============

def face_normals(vertices, faces):
    # vertices: (1, V, 3), faces: (1, F, 3)
    v0 = vertices[:, faces[0, :, 0], :]
    v1 = vertices[:, faces[0, :, 1], :]
    v2 = vertices[:, faces[0, :, 2], :]
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    normals = F.normalize(normals, dim=-1)
    return normals

def normal_loss(pred_vert, target_vert, face):
    """
    Compute cosine similarity loss between face normals.
    """
    pred_normals = face_normals(pred_vert, face)
    target_normals = face_normals(target_vert, face)
    cosine = (pred_normals * target_normals).sum(dim=-1)
    return (1 - cosine).mean()  # smaller is better

def chamfer_distance_naive(x, y):
    """
    Naive Chamfer distance 
    """
    x = x.squeeze(0)  # (N1, 3)
    y = y.squeeze(0)  # (N2, 3)
    diff_xy = torch.cdist(x, y)  # (N1, N2)
    diff_yx = diff_xy.t()        # (N2, N1)
    return diff_xy.min(dim=1)[0].mean() + diff_yx.min(dim=1)[0].mean()

def random_sample_points(points, num_sample=5000):
    """
    从 (B, N, 3) 点云中随机采样 num_sample 个点，避免内存爆炸。
    """
    B, N, _ = points.shape
    if N <= num_sample:
        return points
    idx = torch.randperm(N)[:num_sample].to(points.device)
    return points[:, idx, :]

def chamfer_distance_naive_batchwise(x, y):
    """
    Chamfer Distance between 2 point sets: x and y, shape (B, N, 3)
    Uses iterative batched pairwise min, avoids large pairwise distance matrices.
    """
    chamfer = 0.0
    B = x.shape[0]
    for b in range(B):
        xb = x[b]  # (N, 3)
        yb = y[b]  # (M, 3)

        # 防止过多显存消耗，分块计算距离
        dists = torch.cdist(xb.unsqueeze(0), yb.unsqueeze(0)).squeeze(0)  # (N, M)
        min_xy = dists.min(dim=1)[0].mean()
        min_yx = dists.min(dim=0)[0].mean()
        chamfer += min_xy + min_yx

    return chamfer / B

# =============模型训练==============
def train(model, dataloader, optimizer, device, init_vert, init_face, out_dir, hemi, surface, num_epochs=100, n_steps=7):

    model.train() 

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in dataloader:
            vol = batch['vol'].to(device)         
            init_vert = init_vert.to(device) 
            target_vert_wm = batch['vert_wm'].to(device)
            init_face = init_face.to(device) 

            optimizer.zero_grad()
            # print('vol.shape: ', vol.shape)

            vol = vol.squeeze(2)  
            # print('vol.shape: ', vol.shape)
            # print('init_vert_wm.shape: ', init_vert.shape)
            # print('target_vert_wm.shape: ', target_vert_wm.shape)
            pred_vert_wm = model(init_vert, vol, n_steps=n_steps)  
            # pred_vert_smoothed = taubin_smooth(pred_vert_wm, face, n_iters=5)  # 加平滑

            if target_vert_wm.shape[1] == 1:
                target_vert_wm = target_vert_wm.squeeze(1)

            edge = torch.cat([
                init_face[0,:,[0,1]],
                init_face[0,:,[1,2]],
                init_face[0,:,[2,0]]], dim=0).T

            # loss_mse = F.mse_loss(pred_vert_wm, target_vert_wm)
            loss_edge = edge_distortion(pred_vert_wm, target_vert_wm, edge)
            # loss_area = area_distortion(pred_vert_wm, target_vert_wm, init_face)
            sampled_pred = random_sample_points(pred_vert_wm, 5000)
            sampled_target = random_sample_points(target_vert_wm, 5000)
            loss_chamfer = chamfer_distance_naive_batchwise(sampled_pred, sampled_target)

            loss_normal = normal_loss(pred_vert_wm, target_vert_wm, init_face)
            # loss_chamfer = chamfer_distance_naive(pred_vert_wm, target_vert_wm)
            loss =  loss_normal * 3.0 + loss_chamfer  + loss_edge * 0.3


            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

        # ========== 每 50 个 epoch 保存一次 ==========
        if (epoch + 1) % 50 == 0:
            model_filename = f"model_hemi-{hemi}_{surface}-{epoch+1}-cd-nc-edge.pt"
            model_path = os.path.join(out_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            print(f"Saved model at {model_path}")



# =============模型训练==============
if __name__ == '__main__':

    
    print('# ============ 加载参数 ============')
    parser = argparse.ArgumentParser(description="dHCP DL surface restruction training Pipeline")
    parser.add_argument('--in_dir', default='./in_dir/', type=str,
                        help='Diectory containing input images.')
    parser.add_argument('--out_dir', default='./out_dir/', type=str,
                        help='Directory for saving the output of the pipeline.')
    parser.add_argument('--restore_suffix', default='_desc_T2w_restore_brain.nii.gz', type=str,
                        help='Suffix of T2 restore image file.')
    parser.add_argument('--hemi', default='left', type=str,
                        help='Training for left or right part of brain')
    parser.add_argument('--surface', default='wm', type=str,
                        help='Training for wm or pial of brain')                   
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device for running the pipeline: [cuda, cpu]')
    parser.add_argument('--verbose', action='store_true',
                        help='Print debugging information.')
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    restore_suffix = args.restore_suffix
    hemi = args.hemi # 'left' or 'right'
    surface = args.surface # 'wm' or 'pial'

    device = args.device
    verbose = args.verbose

    max_regist_iter = 5
    min_regist_dice = 0.9
    print('# ============ 加载参数Finished ============')

    
    print('# ============ 模板导入 =============')
    img_t2_atlas_ants = ants.image_read(
        './template/dhcp_week-40_template_T2w.nii.gz')
    # both ants->nibabel and nibabel->ants need to reload the nifiti file
    # so here simply load the image again
    affine_t2_atlas = nib.load(
        './template/dhcp_week-40_template_T2w.nii.gz').affine

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
    print('# ============ 模板导入Finished =============')


    print('# ============ 表面重建架构导入 =============')
    nn_surf_left_wm = SurfDeform(
        C_hid=[8,16,32,64,128,128], C_in=1, sigma=1.0, device=device)
    nn_surf_right_wm = SurfDeform(
        C_hid=[8,16,32,64,128,128], C_in=1, sigma=1.0, device=device)
    nn_surf_left_pial = SurfDeform(
        C_hid=[8,16,32,32,32,32], C_in=1, sigma=1.0, device=device)
    nn_surf_right_pial = SurfDeform(
        C_hid=[8,16,32,32,32,32], C_in=1, sigma=1.0, device=device)
    print('# ============ 表面重建架构导入Finished =============')
    


    print('# ============ 加载对应模板 =============')
    if hemi == 'left':
        vert_in = vert_left_in
        face_in = face_left_in
        if surface == 'wm':
            model = nn_surf_left_wm
        elif surface == 'pial':
            model = nn_surf_left_pial
        print('# ============ 加载对应模板Finished =============')   
    elif hemi == 'right':
        print('Finished add the template')
        vert_in = vert_right_in
        face_in = face_right_in
        if surface == 'wm':
            model = nn_surf_right_wm
        elif surface == 'pial':
            model = nn_surf_right_pial  
        print('# ============ 加载对应模板Finished =============')

    data_pairs = load_dataset(in_dir, surf_hemi='left')
    dataset = SurfaceDataset(data_pairs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train(
            model = model,
            dataloader = dataloader,         
            optimizer = optimizer,
            device = device,
            init_vert = vert_in,
            init_face = face_in,                       
            
            out_dir = out_dir,
            hemi = hemi,
            surface = surface,
            num_epochs = 200,
            n_steps = 7
        )

    # model_filename = f"model_hemi-{hemi}_{surface}-200-cd-nc-edge.pt"
    # model_path = os.path.join(out_dir, model_filename)

    # 保存模型
    # torch.save(model.state_dict(), model_path)
    