import os

'''
用于筛出942个HCP数据集当中有问题的2个
'''
# 数据集路径
hcp_root = '/root/autodl-tmp/hcp1200_dataset/HCP1200'

# 所需的七个文件名模板（每个子文件夹里应有这些文件）
required_files = [
    '{id}.L.pial.native.surf.gii',
    '{id}.L.white.native.surf.gii',
    '{id}.R.pial.native.surf.gii',
    '{id}.R.white.native.surf.gii',
    'brainmask_fs.nii.gz',
    'T1w.nii.gz',
    'T2w.nii.gz',
]

# 找出缺少任何必需文件的子目录
missing_list = []

for subj_id in os.listdir(hcp_root):
    subj_path = os.path.join(hcp_root, subj_id)
    if not os.path.isdir(subj_path):
        continue

    missing = False
    for template in required_files:
        expected_path = os.path.join(subj_path, template.format(id=subj_id))
        if not os.path.isfile(expected_path):
            missing = True
            break

    if missing:
        missing_list.append(subj_id)

# 输出缺失列表
print(f"共有 {len(missing_list)} 个子文件夹缺少所需文件：")
for m in missing_list:
    print(m)
