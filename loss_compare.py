
import re
import matplotlib.pyplot as plt

# 日志文件路径和标签
log_paths = [
    "/root/autodl-tmp/hcp1200/surface/ckpts_all_t2/log_hemi-left_wm_0001.log",
    "/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_5_1/log_hemi-left_wm_0001.log",
    "/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_monai_03/log_hemi-left_wm_0001.log",
    "/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_monai_01/log_hemi-0.1left_wm_0003.log",
    "/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_repeat_5_1/log_hemi-left_wm_0001.log",
    "/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_soft_repeat_5_1/log_hemi-left_wm_0001.log"

]
labels = ['T2w','T1w+T2w', 'T1w+T2w+30%MONAI', 'T1w+T2w+10%MONAI', 'T1w+T2w+Loss_repeat', 'T1w+T2w+Loss_soft_repeat']
colors = ['yellow', 'red', 'black', 'blue', 'purple','green']



# 用于保存每个日志文件中的训练loss
all_train_losses = []

# 正则表达式提取训练loss
pattern = re.compile(r"epoch:(\d+), loss:([0-9.]+)")

for path in log_paths:
    with open(path, 'r') as f:
        log = f.read()

    matches = pattern.findall(log)
    epochs = [int(m[0]) for m in matches]
    losses = [float(m[1]) for m in matches]
    all_train_losses.append((epochs, losses))

# 绘图
plt.figure(figsize=(10, 5))

for i, (epochs, losses) in enumerate(all_train_losses):
    plt.plot(epochs, losses, label=labels[i], color=colors[i])

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison Across Logs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.ylim(0, 20)
# 保存图片而不是显示
plt.savefig("train_loss_comparison.png", dpi=300)

'''

# 正则模式
epoch_pattern = re.compile(r'epoch:(\d+)')
recon_pattern = re.compile(r'recon error:([0-9.]+)')

# 提取每个日志中的 recon error
all_valid_recons = []

for path in log_paths:
    with open(path, 'r') as f:
        lines = f.readlines()

    epochs = []
    recons = []

    for i in range(len(lines)):
        if '------------ validation ------------' in lines[i]:
            # 找到接下来的 epoch 行和 recon 行
            if i+2 < len(lines):
                epoch_match = epoch_pattern.search(lines[i+1])
                recon_match = recon_pattern.search(lines[i+2])
                if epoch_match and recon_match:
                    epochs.append(int(epoch_match.group(1)))
                    recons.append(float(recon_match.group(1)))

    all_valid_recons.append((epochs, recons))

# 绘图
plt.figure(figsize=(6, 6))

for i, (epochs, recons) in enumerate(all_valid_recons):
    plt.plot(epochs, recons, label=labels[i], color=colors[i], marker='o')

plt.xlabel("Epoch")
plt.ylabel("Validation Recon Error")
plt.title("Validation Recon Error Comparison Across Logs")
plt.legend()
plt.grid(True)
# plt.ylim(0.85, 1.5)
plt.tight_layout()
plt.savefig("valid_recon_comparison.png", dpi=300)
# plt.savefig("valid_recon_comparison_focus.png", dpi=300)

'''