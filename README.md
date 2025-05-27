# hcp_1200

## 数据集说明
- [x] hcp1200_dataset/HCP1200: HCP原始数据，包括T1w、T2w、mask、rh与lh的wm与pial
- [x] hcp1200_dataset/HCP1200_cut: HCP原始数据10组cut
- [x] hcp1200_dataset/HCP1200_cut_split: 经过预处理的HCP原始数据10组cut

对比HCP1200和dhcp的template

1.在run_pipeline里
```python
# HCP1200
img_t2_atlas_ants = ants.image_read('./template/dhcp_week-40_template_T2w.nii.gz')
affine_t2_atlas = nib.load('./template/dhcp_week-40_template_T2w.nii.gz').affine
surf_left_in = nib.load('./template/dhcp_week-40_hemi-left_init.surf.gii')
surf_right_in = nib.load('./template/dhcp_week-40_hemi-right_init.surf.gii')

```
```python
# dhcp
img_t1_atlas_ants = ants.image_read('/root/dhcp/hcp_template/MNI152_T1_0.7mm_brain_sampled.nii.gz')
affine_t1_atlas = nib.load('/root/dhcp/hcp_template/MNI152_T1_0.7mm_brain_sampled.nii.gz').affine 
surf_left_in = nib.load('/root/dhcp/hcp_template/lh.white.surf.gii')
surf_right_in = nib.load('/root/dhcp/hcp_template/rh.white.surf.gii')
```

2.在train
```python
class SurfDataset(Dataset):
    # ------ 加载模板数据 ------
    file_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(file_dir, '../hcp_template/')

    # 1. 加载模板体积
    self.img_temp = nib.load(os.path.join(template_dir, 'MNI152_T1_0.7mm_brain_sampled.nii.gz'))
    self.affine_temp = self.img_temp.affine
    # 2. 加载模板表面
    surf_temp = nib.load(
        os.path.join(template_dir, 
        f"{'lh' if self.surf_hemi == 'left' else 'rh'}."
        f"{'white' if self.surf_type == 'wm' else 'pial'}.surf.gii")
    )

def train_loop(args):

    # ------ pre-compute adjacency------
    file_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(file_dir, '../hcp_template/')
        

    surf_temp = nib.load(
            os.path.join(template_dir, 
            f"{'lh' if surf_hemi == 'left' else 'rh'}."
            f"{'white' if surf_type == 'wm' else 'pial'}.surf.gii")
    )
    face_temp = surf_temp.agg_data('triangle')[:,[2,1,0]]
    face_in = torch.LongTensor(face_temp[None]).to(device)
```

3.模板降采样、预处理环节(同2)
```python
def remesh_template_surfaces(template_dir, n_target=200000):
    """
    将模板表面重采样到指定顶点数量
    Args:
        template_dir: 模板文件目录路径
        n_target: 目标顶点数 (默认15万)
    """
    surf_files = [
        # 'lh.white.surf.gii',
        # 'lh.pial.surf.gii', 
        # 'rh.white.surf.gii',
        # 'rh.pial.surf.gii'
        'S1200.L.inflated_MSMAll.32k_fs_LR_150k.surf.gii',
        # 'S1200.L.very_inflated_MSMAll.32k_fs_LR.surf.gii',
        'S1200.R.inflated_MSMAll.32k_fs_LR_150k.surf.gii',
        # 'S1200.R.very_inflated_MSMAll.32k_fs_LR.surf.gii',
    ]
```

4.平滑表面生成(同2)
```python
wb_generate_inflated_surfaces('/root/dhcp/hcp_template/lh.white.surf.gii')
```

运行代码
```bash
nohup python3 train_seg_brain.py > train_seg_brain.log 2>&1 &
```