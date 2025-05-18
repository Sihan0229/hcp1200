import re
import matplotlib.pyplot as plt

def parse_log_file(log_path):
    train_losses = {}
    val_losses = {}

    with open(log_path, 'r') as f:
        lines = f.readlines()

    val_epoch_buffer = None  # 用于临时保存验证时的 epoch

    for line in lines:
        # 提取训练损失
        match_train = re.search(r"epoch:(\d+), loss:([\d\.]+)", line)
        if match_train:
            epoch = int(match_train.group(1))
            loss = float(match_train.group(2))
            train_losses[epoch] = loss

        # 识别验证阶段 epoch
        match_val_epoch = re.search(r"epoch:(\d+)", line)
        if match_val_epoch:
            val_epoch_buffer = int(match_val_epoch.group(1))

        # 如果有 recon error，就与之前的 epoch 对应
        match_val_loss = re.search(r"recon error:([\d\.]+)", line)
        if match_val_loss and val_epoch_buffer is not None:
            loss = float(match_val_loss.group(1))
            val_losses[val_epoch_buffer] = loss
            val_epoch_buffer = None  # 清空 buffer，防止错配

    return train_losses, val_losses

def plot_losses(train_losses, val_losses, save_path='loss_curve.png'):
    epochs_train = sorted(train_losses.keys())
    losses_train = [train_losses[e] for e in epochs_train]

    epochs_val = sorted(val_losses.keys())
    losses_val = [val_losses[e] for e in epochs_val]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_train, losses_train, label='Training Loss', color='blue')
    plt.plot(epochs_val, losses_val, label='Validation Loss', color='red', marker='o', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs (T1 + T2w)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved loss plot to: {save_path}")

# 用法
log_file_path = '/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_5_1/log_hemi-left_wm_0001.log'
save_img_path = '/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_5_1/log_hemi-left_wm_0001.png'

train_losses, val_losses = parse_log_file(log_file_path)
plot_losses(train_losses, val_losses, save_path=save_img_path)
