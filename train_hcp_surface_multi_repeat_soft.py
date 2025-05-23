import sys
import os
# os.chdir('..')
# sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm
import nibabel as nib
import glob
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points

from surface.net import SurfDeform

from utils.mesh import (
    apply_affine_mat,
    adjacent_faces,
    taubin_smooth,
    face_normal)


def density_loss_pytorch3d(pred_points, gt_points, k=30, sigma=0.05):
    """
    pred_points: (B, N_pred, 3)
    gt_points: (B, N_gt, 3)
    k: number of neighbors to consider for KDE
    sigma: bandwidth of Gaussian kernel

    Returns scalar loss tensor
    """
    # 找每个 gt 点的 k 近邻 pred 点
    knn = knn_points(gt_points, pred_points, K=k, return_nn=False)
    # knn.dists: (B, N_gt, k), 欧式距离平方

    # 计算 Gaussian 权重
    weights = torch.exp(-knn.dists / (2 * sigma ** 2))  # (B, N_gt, k)

    # 计算每个 GT 点的局部密度（邻居权重和）
    density_per_gt = weights.sum(dim=2)  # (B, N_gt)

    expected_density = 1.0
    loss = ((density_per_gt - expected_density) ** 2).mean()

    return loss

def chamfer_with_match_count(pred, gt):
    """
    Inputs:
    - pred: (B, N, 3) 预测点集
    - gt: (B, M, 3) 真实点集
    Returns:
    - chamfer_loss: scalar
    - match_counts: list[(M,) tensor] 每个 batch 中真实点被匹配的次数
    """

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

class SurfDataset(Dataset):
    """
    Dataset class for surface reconstruction (Lazy Loading Version)
    """
    def __init__(self, args, data_split='train'):
        super(SurfDataset, self).__init__()
        
        # ------ 加载固定参数 ------ 
        self.surf_hemi = args.surf_hemi
        self.surf_type = args.surf_type
        self.device = args.device
        self.sigma = args.sigma

        # ------ 获取数据路径列表 ------
        # self.subj_dirs = sorted(glob.glob(f'/root/autodl-tmp/hcp1200_dataset/HCP1200_split/{data_split}/*'))
        
        logging.info("load dataset ...")

        if data_split == 'train':
            fold_ids = [0, 1, 2, 3]  # 合并4个fold作为训练集
        elif data_split == 'valid':
            fold_ids = [4]  # 单独1个fold作为验证集
        else:
            raise ValueError(f"Unsupported data_split: {data_split}")

        self.subj_dirs = []
        for fold_id in fold_ids:
            fold_path = f'/root/autodl-tmp/hcp1200_dataset/HCP1200_split/train_valid/fold_{fold_id}/*'
            self.subj_dirs.extend(sorted(glob.glob(fold_path)))

        logging.info(f"Loaded {len(self.subj_dirs)} samples from folds {fold_ids} for {data_split} set.")

        # ------ 加载模板数据 ------
        file_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(file_dir, '/root/autodl-tmp/hcp1200/template_hcp1200/')
        
        # 1. 加载模板体积
        self.img_temp = nib.load(os.path.join(template_dir, 'MNI152_T1_0.7mm_brain_sampled.nii.gz'))
        self.affine_temp = self.img_temp.affine
        # 2. 加载模板表面
        surf_temp = nib.load(
            os.path.join(template_dir, 
            f"{'lh' if self.surf_hemi == 'left' else 'rh'}."
            f"{'white' if self.surf_type == 'wm' else 'pial'}.surf.gii")
        )
        # surf_temp = nib.load(
        #     os.path.join(template_dir, 
        #     "S1200."
        #     f"{'L' if self.surf_hemi == 'left' else 'R'}."
        #     "inflated_MSMAll.32k_fs_LR_150k.surf.gii")
        # )
        self.vert_temp = apply_affine_mat(
            surf_temp.agg_data('pointset'),
            np.linalg.inv(self.affine_temp)
        )
        self.face_temp = surf_temp.agg_data('triangle')[:, [2,1,0]]
        
        # ------ 预加载模型（仅pial需要）------
        self.nn_surf = None
        if self.surf_type == 'pial':
            self.nn_surf = SurfDeform(
                C_hid=[8,16,32,64,128,128], C_in=1,
                inshape=[160,304,256], sigma=self.sigma, device=self.device)
            model_path = f'/root/dhcp/train/surface/ckpts/model_hemi-left_wm_0004_100epochs.pt'
            self.nn_surf.load_state_dict(torch.load(model_path, map_location=self.device))

    def __len__(self):
        return len(self.subj_dirs)

    def __getitem__(self, idx):
        subj_dir = self.subj_dirs[idx]
        subj_id = os.path.basename(subj_dir)
        
        
        # ------ 惰性加载体积数据（T2w + T1w）------
        t2_img = nib.load(os.path.join(subj_dir, 'T2w_proc_affine.nii.gz'))
        affine_in = t2_img.affine
        t2_data = t2_img.get_fdata()
        t2_data = (t2_data / t2_data.max()).astype(np.float32)

        t1_img = nib.load(os.path.join(subj_dir, 'T1w_proc_affine.nii.gz'))
        t1_data = t1_img.get_fdata()
        t1_data = (t1_data / t1_data.max()).astype(np.float32)

        # 裁剪半球（分别对 T1w/T2w）
        if self.surf_hemi == 'left':
            t2_data = t2_data[96:]
            t1_data = t1_data[96:]
        elif self.surf_hemi == 'right':
            t2_data = t2_data[:160]
            t1_data = t1_data[:160]

        # 合并为 2 通道体积 [2, H, W, D]
        vol_data = np.stack([t1_data, t2_data], axis=0)  # [2, H, W, D]


        # ------ 处理表面数据 ------
        vert_in = self.vert_temp.copy().astype(np.float32)
        face_in = self.face_temp.copy()
        
        if self.surf_hemi == 'left':
            vert_in[:, 0] -= 96
            
        # pial表面需要预测
        if self.surf_type == 'pial':
            vert_in = torch.Tensor(vert_in[None]).to(self.device)
            face_in = torch.LongTensor(face_in[None]).to(self.device)
            vol_tensor = torch.Tensor(vol_data[None]).to(self.device)
            
            with torch.no_grad():
                vert_in = self.nn_surf(vert_in, vol_tensor, n_steps=7)
                vert_in = taubin_smooth(vert_in, face_in, n_iters=5)
            
            vert_in = vert_in[0].cpu().numpy()
            face_in = face_in[0].cpu().numpy()
            vol_data = vol_tensor[0].cpu().numpy()
        
        # ------ 惰性加载GT表面 ------
        surf_gt = nib.load(
            os.path.join(subj_dir, 
            f"{subj_id}.{'L' if self.surf_hemi == 'left' else 'R'}."
            f"{'white' if self.surf_type == 'wm' else 'pial'}.native_160k.surf.gii")
        )
        vert_gt = apply_affine_mat(
            surf_gt.agg_data('pointset'),
            np.linalg.inv(affine_in)
        ).astype(np.float32)
        face_gt = surf_gt.agg_data('triangle')[:, [2,1,0]]
        
        if self.surf_hemi == 'left':
            vert_gt[:, 0] -= 96
            
        return (vol_data, vert_in, vert_gt, face_in, face_gt)
    
def train_loop(args):
    # ------ load arguments ------ 
    surf_type = args.surf_type  # wm or pial
    surf_hemi = args.surf_hemi  # left or right
    tag = args.tag
    device = torch.device(args.device)
    n_epoch = args.n_epoch  # training epochs
    lr = args.lr  # learning rate
    sigma = args.sigma  # std for gaussian filter
    w_nc = args.w_nc  # weight for nc loss
    w_edge = args.w_edge  # weight for edge loss
    w_repeat = args.w_repeat  # weight for edge loss

    
    # start training logging
    # logging.basicConfig(
    #     filename='/root/autodl-tmp/hcp1200/surface/ckpts/log_hemi-'+surf_hemi+'_'+\
    #     surf_type+'_'+tag+'.log', level=logging.INFO,
    #     format='%(asctime)s %(message)s')
    
    logging.basicConfig( 
    filename='/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_repeat_soft_5_1/log_hemi-' + surf_hemi + '_' + 
             surf_type + '_' + tag + '.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s')

    # ------ load dataset ------ 
    logging.info("load dataset ...")
    trainset = SurfDataset(args, data_split='train')
    validset = SurfDataset(args, data_split='valid')

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=16)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=16)
    
    # ------ pre-compute adjacency------
    file_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(file_dir, '/root/autodl-tmp/hcp1200/template_hcp1200/')
    
    surf_temp = nib.load(
        os.path.join(template_dir, 
        f"{'lh' if surf_hemi == 'left' else 'rh'}."
        f"{'white' if surf_type == 'wm' else 'pial'}.surf.gii")
    )
    face_temp = surf_temp.agg_data('triangle')[:,[2,1,0]]
    face_in = torch.LongTensor(face_temp[None]).to(device)
    # for normal consistency loss
    adj_faces = adjacent_faces(face_in)
    # for edge length loss
    edge_in = torch.cat([face_in[0,:,[0,1]],
                         face_in[0,:,[1,2]],
                         face_in[0,:,[2,0]]], dim=0).T

    # ------ initialize model ------ 
    logging.info("initalize model ...")
    if surf_type == 'wm':
        C_hid = [8,16,32,64,128,128]  # number of channels for each layer
    elif surf_type == 'pial':
        C_hid = [8,16,32,32,32,32]  # fewer params to avoid overfitting
    # if surf_type == 'wm': # TODO
    #     C_hid = [16,32,64,128,256,256]  # number of channels for each layer
    # elif surf_type == 'pial':
    #     C_hid = [16,32,64,64,64,64]  # fewer params to avoid overfitting
    # nn_surf = SurfDeform(
    #     C_hid=C_hid, C_in=1, inshape=[112,224,160],
    #     sigma=sigma, device=device)
    nn_surf = SurfDeform(
        C_hid=C_hid, C_in=2, inshape=[160,304,256],
        sigma=sigma, device=device)
    optimizer = optim.Adam(nn_surf.parameters(), lr=lr)

    # ------ training loop ------ 
    accumulation_steps = 4  # 例如，每4步更新一次梯度
    logging.info("start training ...")
    for epoch in tqdm(range(n_epoch+1)):
        avg_loss = []
        optimizer.zero_grad()  # 初始化放到外面
        for idx, data in enumerate(trainloader):
            vol_in, vert_in, vert_gt, face_in, face_gt = data
            vol_in = vol_in.to(device).float()
            vert_in = vert_in.to(device).float()
            face_in = face_in.to(device).long()
            vert_gt = vert_gt.to(device).float()
            face_gt = face_gt.to(device).long()
            # optimizer.zero_grad()
            vert_pred = nn_surf(vert_in, vol_in, n_steps=7) # TODO

            # normal consistency loss
            normal = face_normal(vert_pred, face_in)  # face normal
            nc_loss = (1 - normal[:,adj_faces].prod(-2).sum(-1)).mean()
            # edge loss
            vert_i = vert_pred[:,edge_in[0]]
            vert_j = vert_pred[:,edge_in[1]]
            edge_loss = ((vert_i - vert_j)**2).sum(-1).mean() 
            # reconstruction loss
            # recon_loss = chamfer_distance(vert_pred, vert_gt)[0]

            # chamfer_with_match_count 返回 loss 和 每个 GT 点被匹配的次数
            # recon_loss, match_counts = chamfer_with_match_count(vert_pred, vert_gt)

            # # 统计每个 batch 中，GT 点被匹配次数 > 1 的点的个数
            # # print(match_counts)
            # match_counts_tensor = match_counts[0]  # 取出列表里的 Tensor
            # repeated_matches = (match_counts_tensor > 1).sum()  # 统计大于1的个数
            # repeat_loss = repeated_matches.float().mean()   # 可作为损失函数值
            

            # loss = recon_loss + w_nc*nc_loss + w_edge*edge_loss + w_repeat * repeat_loss 
            # loss = loss / accumulation_steps  # 注意缩放 loss

            # reconstruction loss（用pytorch3d的Chamfer距离或其他）
            recon_loss = chamfer_distance(vert_pred, vert_gt)[0]

            # 软重复匹配惩罚
            # repeat_loss = soft_repeat_loss(vert_pred, vert_gt, temperature=0.01)
            w_density = 0.1
            density_reg_loss = density_loss_pytorch3d(vert_pred, vert_gt, k=30, sigma=0.05)
            loss = recon_loss + w_nc * nc_loss + w_edge * edge_loss + w_density * density_reg_loss

            # 总损失
            # loss = recon_loss + w_nc * nc_loss + w_edge * edge_loss + w_repeat * repeat_loss
            loss = loss / accumulation_steps


            # avg_loss.append(loss.item())
            
            loss.backward()

            avg_loss.append(loss.item() * accumulation_steps)  # 乘回来表示真实 loss

            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(trainloader):
                optimizer.step()
                optimizer.zero_grad()
            # optimizer.step()
            # print(f'sample {idx}: loss = {loss.item()}, average loss = {np.mean(avg_loss)}')
       
        logging.info("epoch:{}, loss:{}".format(epoch, np.mean(avg_loss)))

        if epoch % 10 == 0:  # start validation
            logging.info('------------ validation ------------')
            with torch.no_grad():
                recon_error = []
                nc_error = []
                edge_error = []
                repeat_error = []
                density_error = []
                for idx, data in enumerate(validloader):
                    vol_in, vert_in, vert_gt, face_in, face_gt = data
                    vol_in = vol_in.to(device).float()
                    vert_in = vert_in.to(device).float()
                    face_in = face_in.to(device).long()
                    vert_gt = vert_gt.to(device).float()
                    face_gt = face_gt.to(device).long()
                    
                    vert_pred = nn_surf(vert_in, vol_in, n_steps=7) # TODO
                    # recon_loss = chamfer_distance(vert_pred, vert_gt)[0]
                    recon_loss, match_counts = chamfer_with_match_count(vert_pred, vert_gt)
                    # 统计每个 batch 中，GT 点被匹配次数 > 1 的点的个数
                    # repeated_matches = (match_counts > 1).float().sum(dim=1)  # shape: [B]
                    # 可选：取 batch 中的平均作为惩罚项（你也可以选择求和）
                    match_counts_tensor = match_counts[0]  # 取出列表里的 Tensor
                    repeated_matches = (match_counts_tensor > 1).sum()  # 统计大于1的个数
                    repeat_loss = repeated_matches.float().mean()   # 可作为损失函数值
                    recon_error.append(recon_loss.item())
                    repeat_error.append(repeat_loss.item())

                    w_density = 0.1
                    density_reg_loss = density_loss_pytorch3d(vert_pred, vert_gt, k=30, sigma=0.05)
                    density_error.append(density_reg_loss.item())



                    normal = face_normal(vert_pred, face_in)  # face normal
                    nc_loss = (1 - normal[:,adj_faces].prod(-2).sum(-1)).mean()
                    nc_error.append(nc_loss.item())

                    # edge loss
                    vert_i = vert_pred[:,edge_in[0]]
                    vert_j = vert_pred[:,edge_in[1]]
                    edge_loss = ((vert_i - vert_j)**2).sum(-1).mean() 
                    edge_error.append(edge_loss.item())



            logging.info('epoch:{}'.format(epoch))
            logging.info('recon error:{}'.format(np.mean(recon_error)))
            logging.info('nc error:{}'.format(np.mean(nc_error)))
            logging.info('edge error:{}'.format(np.mean(edge_error)))
            logging.info('repeat error:{}'.format(np.mean(repeat_error)))
            logging.info('density error:{}'.format(np.mean(density_error)))


            logging.info('-------------------------------------')
        
            # save model checkpoints
            torch.save(nn_surf.state_dict(),
                       '/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_repeat_soft_5_1/model_hemi-'+surf_hemi+'_'+\
                       surf_type+'_'+tag+'_'+str(epoch)+'epochs.pt')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Surface Recon")
    
    parser.add_argument('--surf_type', default='wm', type=str, help="[wm, pial]")
    parser.add_argument('--surf_hemi', default='left', type=str, help="[left, right]")
    parser.add_argument('--device', default="cuda", type=str, help="[cuda, cpu]")
    parser.add_argument('--tag', default='0001', type=str, help="identity for experiments")

    # parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    # parser.add_argument('--n_epoch', default=200, type=int, help="number of training epochs")
    # parser.add_argument('--sigma', default=1.0, type=float, help="standard deviation for gaussian smooth")
    # parser.add_argument('--w_nc', default=3.0, type=float, help="weight for normal consistency loss")
    # parser.add_argument('--w_edge', default=0.3, type=float, help="weight for edge length loss")
    
    parser.add_argument('--lr', default=1e-4 , type=float, help="learning rate") # TODO # 更小的学习率适应复杂结构
    parser.add_argument('--n_epoch', default=300, type=int, help="number of training epochs") # TODO # 延长训练周期
    parser.add_argument('--sigma', default=0.7, type=float, help="standard deviation for gaussian smooth") # TODO # 减少平滑以保留颞叶、顶叶等区域的细微褶皱
    parser.add_argument('--w_nc', default=3.0, type=float, help="weight for normal consistency loss") # TODO # 降低对平滑的强约束
    parser.add_argument('--w_edge', default=0.3, type=float, help="weight for edge length loss") # TODO # 增强边缘保持（成人皮质更薄）
    parser.add_argument('--w_repeat', default=0.001, type=float, help="weight for edge length loss") # TODO # 增强边缘保持（成人皮质更薄）
    
    args = parser.parse_args()
    
    train_loop(args)