import os

'''
用来把影响T2w读取的文件改名
'''

# 要处理的根目录
root_dir = "/root/autodl-tmp/hcp_gcn/datasets_test"

# 遍历所有子目录和文件
for subdir, _, files in os.walk(root_dir):
    for filename in files:
        old_path = os.path.join(subdir, filename)
        
        # 修改 _desc-restore_T2w.nii.gz 为 _desc_T2w_restore_brain.nii.gz
        if filename.endswith("_desc-restore_T2w.nii.gz"):
            new_filename = filename.replace("_desc-restore_T2w.nii.gz", "_desc_T2w_restore_brain.nii.gz")
            new_path = os.path.join(subdir, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} --> {new_path}")
        
        # 修改 _desc-restore_T1w.nii.gz 为 _desc_T1w_restore_brain.nii.gz
        elif filename.endswith("_desc-restore_T1w.nii.gz"):
            new_filename = filename.replace("_desc-restore_T1w.nii.gz", "_desc_T1w_restore_brain.nii.gz")
            new_path = os.path.join(subdir, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} --> {new_path}")

        elif filename.endswith("_desc-restore_T2w.json"):
            new_filename = filename.replace("_desc-restore_T2w.json", "_desc_T2w_restore_brain.json")
            new_path = os.path.join(subdir, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} --> {new_path}")

        elif filename.endswith("_desc-restore_T1w.json"):
            new_filename = filename.replace("_desc-restore_T1w.json", "_desc_T1w_restore_brain.json")
            new_path = os.path.join(subdir, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} --> {new_path}")

