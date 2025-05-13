import os
import sys
from glob import glob
import time
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# 获取当前文件的父目录
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 将父目录的同级目录 seg 添加到 Python 路径
sys.path.append(parent_dir)

from unet import UNet

# 数据加载器
class BrainDataset(Dataset):
    def __init__(self, t1w_paths, mask_paths):
        self.t1w_paths = t1w_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.t1w_paths)

    def __getitem__(self, idx):
        # 加载 T1w 图像
        t1w = nib.load(self.t1w_paths[idx]).get_fdata() # type: ignore
        t1w = torch.tensor(t1w, dtype=torch.float32).unsqueeze(0)
        t1w = F.interpolate(t1w.unsqueeze(0), size=[256, 304, 256], mode='trilinear').squeeze(0)
        t1w = t1w / t1w.max()

        # 加载 brain_mask
        mask = nib.load(self.mask_paths[idx]).get_fdata() # type: ignore
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return t1w, mask

# 数据目录
data_dir = "/home/sywang/datasets"

# 获取所有被试的 T1w 和 brainmask_fs 文件路径
t1w_paths = sorted(glob(os.path.join(data_dir, "*", "T1w.nii.gz")))
mask_paths = sorted(glob(os.path.join(data_dir, "*", "brainmask_fs.nii.gz")))

# 检查数据是否匹配
assert len(t1w_paths) == len(mask_paths), "not match"
print(f"{len(t1w_paths)} subjects")

# 随机打乱数据
indices = torch.randperm(len(t1w_paths)).tolist()
t1w_paths = [t1w_paths[i] for i in indices]
mask_paths = [mask_paths[i] for i in indices]

# 划分训练集、验证集和测试集（80% 训练集，10% 验证集，10% 测试集）
train_t1w, test_t1w, train_mask, test_mask = train_test_split(t1w_paths, mask_paths, test_size=0.2, random_state=42)
val_t1w, test_t1w, val_mask, test_mask = train_test_split(test_t1w, test_mask, test_size=0.5, random_state=42)

# 创建 DataLoader
train_dataset = BrainDataset(train_t1w, train_mask)
val_dataset = BrainDataset(val_t1w, val_mask)
test_dataset = BrainDataset(test_t1w, test_mask)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# TODO: 这里需要调整torch版本，不行就重建虚拟环境
# 训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 初始化模型
model = UNet(
    C_in=1, C_hid=[16,32,32,32,32], C_out=1).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 模型训练
num_epochs = 100
model.to(device)

best_val_loss = float('inf')

for epoch in range(num_epochs):
    
    epoch_start_time = time.time() 

    model.train()
    train_loss = 0.0
    
    batch_start_time = time.time()
    
    for batch_idx, (t1w, mask) in enumerate(train_dataloader):
        
        t1w, mask = t1w.to(device), mask.to(device)

        t1w = F.interpolate(t1w, size=(256, 304, 256), mode='trilinear')
        mask = F.interpolate(mask, size=(256, 304, 256), mode='trilinear')
        

        # print(f"Batch {batch_idx}: t1w shape={t1w.shape}, mask shape={mask.shape}")
        
        # 前向传播
        output = model(t1w)
        loss = criterion(output, mask)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
       # 打印训练损失和 batch 用时
        if batch_idx % 10 == 0:
            batch_time = time.time() - batch_start_time  # 计算 batch 用时
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Time: {batch_time:.2f}s")
            batch_start_time = time.time()

    # 计算 epoch 用时
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch [{epoch+1}/{num_epochs}] completed, Time: {epoch_time:.2f}s")
    
    # 验证集评估
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for t1w, mask in val_dataloader:
            t1w, mask = t1w.to(device), mask.to(device)
            t1w = F.interpolate(t1w, size=(256, 304, 256), mode='trilinear')
            mask = F.interpolate(mask, size=(256, 304, 256), mode='trilinear')  
            output = model(t1w)
            val_loss += criterion(output, mask).item()

    val_loss /= len(val_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_save_path = f"../model/model_seg_brain_epoch{epoch}.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Weights have been saved to {model_save_path}")


# 测试集评估
model.eval()
test_loss = 0.0
with torch.no_grad():
    for t1w, mask in test_dataloader:
        t1w, mask = t1w.to(device), mask.to(device)
        
        t1w = F.interpolate(t1w, size=(256, 304, 256), mode='trilinear')
        mask = F.interpolate(mask, size=(256, 304, 256), mode='trilinear')  
        
        output = model(t1w)
        test_loss += criterion(output, mask).item()

test_loss /= len(test_dataloader)
print(f"Test Loss: {test_loss:.4f}")