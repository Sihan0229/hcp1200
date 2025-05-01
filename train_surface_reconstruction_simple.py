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


def train(model, dataloader, optimizer, device, init_vert, out_dir, hemi, surface, num_epochs=100, n_steps=7):

    model.train()  # 进入训练模式

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in dataloader:
            vol = batch['vol'].to(device)         
            init_vert = init_vert.to(device) 
            target_vert_wm = batch['vert_wm'].to(device)
            # face = batch['face_wm'].to(device)

            optimizer.zero_grad()

            vol = vol.squeeze(2)  
            pred_vert_wm = model(init_vert, vol, n_steps=n_steps)  # 模型输出
            # pred_vert_smoothed = taubin_smooth(pred_vert_wm, face, n_iters=5)  # 加平滑

            if target_vert_wm.shape[1] == 1:
                target_vert_wm = target_vert_wm.squeeze(1)

            loss = F.mse_loss(pred_vert_wm, target_vert_wm) 


            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")



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
            
            out_dir = out_dir,
            hemi = hemi,
            surface = surface,
            num_epochs = 500,
            n_steps = 7
        )

    model_filename = f"model_hemi-{hemi}_{surface}-500.pt"
    model_path = os.path.join(out_dir, model_filename)

    # 保存模型
    torch.save(model.state_dict(), model_path)
    