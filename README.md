# hcp_1200

## 数据集说明
+ datasets_dhcp_cut2: 2组T2w原始数据
+ datasets_dhcp_cut50: 50组T2w原始数据
+ datasets for surface resconstruction testing: 2组经过预处理的vol、wm、pial数据，用于皮层重建测试
+ datasets for surface_resconstruction training: 50组经过预处理的vol、wm、pial数据，用于皮层重建训练
+ datasets test:  2组dHCP原始数据，包括T2w和各阶段的Ground Truth
+ datasets train:  50组dHCP原始数据，包括T2w和各阶段的Ground Truth

## 结果文件说明
+ model_trained by_10: 由10组数据训练得到的模型
+ model_trained by_50: 由50组数据训练得到的模型 
+ results_by_my_model: 由本项目模型处理的结果

## py与sh文件说明
+ compare_brain_extraction.py: ROI提取测试对比
+ compare_brain_extraction.sh: ROI提取测试对比
+ compare_ribbon_seg.py: 皮层带分割测试对比（未完成）
+ compare_surface_reconstruction_simple.py: 皮层重建（简化版）损失函数为MSE
+ compare surface reconstruction.py: 皮层重建对比（改进版）损失函数为cd nc edge
+ compare surface reconstruction.sh: 皮层重建对比（改进版）损失函数为cd nc edge

+ train_seg_brain.py: ROI提取模型训练
+ train_surface_reconstruction_simple.py: 皮层重建（MSE）模型训练
+ train_surface_reconstruction_simple.sh: 皮层重建（MSE）模型训练
+ train_surface_reconstruction_wm.py: 皮层白质重建（Loss改进）模型训练
+ train_surface_reconstruction_wm.sh: 皮层白质重建（Loss改进）模型训练
+ train_surface_reconstruction_pial.py: 皮层软脑膜重建（Loss改进）模型训练
+ train_surface_reconstruction_pial.sh: 皮层软脑膜重建（Loss改进）模型训练

+ train.sh: 完整训练流程（未完成）
+ train_pipeline.py: 完整训练流程（未完成）
+ mesh_check.py: 重建表面格式正确性检验
+ mesh reduce.py: 表面降采样
+ T12w_others_rename.py: restore等文件重命名

+ install.sh: 环境配置
+ run_pipeline.py: 原始测试管道
+ run.sh: 原始测试管道


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