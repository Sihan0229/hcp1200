import os
import random
import shutil

# 原始目录
base_dir = "/root/autodl-tmp/hcp1200_dataset/HCP1200_split"

# 获取所有子文件夹名（不包含非目录文件）
all_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
all_folders.sort()  # 可选：排序保证稳定性
random.seed(42)  # 设置随机种子以保证可复现
random.shuffle(all_folders)

# 划分9:1为 train_valid 和 test
num_total = len(all_folders)
num_test = int(num_total * 0.1)
test_folders = all_folders[:num_test]
train_valid_folders = all_folders[num_test:]

# 创建目标文件夹
test_dir = os.path.join(base_dir, "test")
train_valid_dir = os.path.join(base_dir, "train_valid")
os.makedirs(test_dir, exist_ok=True)
os.makedirs(train_valid_dir, exist_ok=True)

# 移动 test 文件夹
for folder in test_folders:
    shutil.move(os.path.join(base_dir, folder), os.path.join(test_dir, folder))

# 移动 train_valid 文件夹
for folder in train_valid_folders:
    shutil.move(os.path.join(base_dir, folder), os.path.join(train_valid_dir, folder))

# 对 train_valid 进行五折划分
folds = [[] for _ in range(5)]
for i, folder in enumerate(sorted(train_valid_folders)):
    folds[i % 5].append(folder)

# 创建每个 fold 文件夹并移动子文件夹
for i, fold in enumerate(folds):
    fold_dir = os.path.join(train_valid_dir, f"fold_{i}")
    os.makedirs(fold_dir, exist_ok=True)
    for folder in fold:
        shutil.move(os.path.join(train_valid_dir, folder), os.path.join(fold_dir, folder))

print(f"总样本数: {num_total}")
print(f"Test集数量: {len(test_folders)}")
print(f"Train_Valid集数量: {len(train_valid_folders)}，每折数量: {[len(f) for f in folds]}")
