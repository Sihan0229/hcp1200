import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from tqdm import tqdm

from seg.unet import UNet  # 你已有的 UNet 实现

# ---------- 数据集 ----------
class BrainSegDataset(Dataset):
    def __init__(self, root_dir, t2_suffix='T2w.nii.gz', mask_suffix='brainmask_fs.nii.gz'):
        self.sample_dirs = sorted(glob.glob(os.path.join(root_dir, '*')))
        self.t2_suffix = t2_suffix
        self.mask_suffix = mask_suffix

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        base = os.path.basename(sample_dir)  # e.g., sub-CC00060XX03_ses-12501

        t2_path = os.path.join(sample_dir, f'{self.t2_suffix}')
        mask_path = os.path.join(sample_dir, f'{self.mask_suffix}')

        # 检查文件是否存在
        if not os.path.exists(t2_path):
            raise FileNotFoundError(f"T2 file not found: {t2_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        t2_img = nib.load(t2_path).get_fdata().astype(np.float32)
        mask_img = nib.load(mask_path).get_fdata().astype(np.float32)

        # Downsample
        t2_tensor = torch.tensor(t2_img).unsqueeze(0).unsqueeze(0)
        mask_tensor = torch.tensor(mask_img).unsqueeze(0).unsqueeze(0)

        t2_tensor = F.interpolate(t2_tensor, size=(160,208,208), mode='trilinear', align_corners=False)
        mask_tensor = F.interpolate(mask_tensor, size=(160,208,208), mode='nearest')

        t2_tensor = t2_tensor / t2_tensor.max()

        return t2_tensor.squeeze(0), mask_tensor.squeeze(0)


# ---------- 训练 ----------
def train_seg_brain(model, dataloader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(n_epoch):
        epoch_loss = 0
        for x, y in tqdm(dataloader, desc=f'Epoch {epoch+1}/{n_epoch}'):
            x = x.to(device)
            y = y.to(device)

            pred = torch.sigmoid(model(x))
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.4f}')

    torch.save(model.state_dict(), '/root/autodl-tmp/hcp1200/result_train_model/model_seg_brain_all.pt')
    print('✅ Model saved to model_seg_brain.pt')


# ---------- 主函数 ----------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_root = '/root/autodl-tmp/hcp1200_dataset/HCP1200' 

    n_epoch = 20
    dataset = BrainSegDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = UNet(C_in=1, C_hid=[16,32,32,32,32], C_out=1).to(device)
    train_seg_brain(model, dataloader, device)
