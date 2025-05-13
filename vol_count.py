import os
import nibabel as nib

# 设置根目录路径
root_dir = '/root/autodl-tmp/hcp_gcn/HCP1200_cut/'

# 遍历根目录下所有子文件夹
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('w.nii.gz'):
            file_path = os.path.join(subdir, file)
            try:
                # 加载 NIfTI 文件
                img = nib.load(file_path)
                shape = img.shape  # 图像尺寸
                print(f"{file_path}: shape = {shape}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
