import re
import matplotlib.pyplot as plt

def parse_log_file(log_path):
    train_losses = {}
    val_rocon_losses = {}
    val_nc_losses = {}
    val_edge_losses = {}
    val_repeat_losses = {}
    val_density_losses = {}
    

    with open(log_path, 'r') as f:
        lines = f.readlines()

    val_epoch_buffer = None

    for line in lines:
        match_train = re.search(r"epoch:(\d+), loss:([\d\.]+)", line)
        if match_train:
            epoch = int(match_train.group(1))
            loss = float(match_train.group(2))
            train_losses[epoch] = loss

        match_val_epoch = re.search(r"epoch:(\d+)$", line.strip())
        if match_val_epoch:
            val_epoch_buffer = int(match_val_epoch.group(1))

        if val_epoch_buffer is not None:
            match_val_rocon_loss = re.search(r"recon error:([\d\.]+)", line)
            if match_val_rocon_loss:
                val_rocon_losses[val_epoch_buffer] = float(match_val_rocon_loss.group(1))

            match_val_nc_loss = re.search(r"nc error:([\d\.]+)", line)
            if match_val_nc_loss:
                val_nc_losses[val_epoch_buffer] = float(match_val_nc_loss.group(1))

            match_val_edge_loss = re.search(r"edge error:([\d\.]+)", line)
            if match_val_edge_loss:
                val_edge_losses[val_epoch_buffer] = float(match_val_edge_loss.group(1))

            match_val_repeat_loss = re.search(r"repeat error:([\d\.]+)", line)
            if match_val_repeat_loss:
                val_repeat_losses[val_epoch_buffer] = float(match_val_repeat_loss.group(1))

            match_val_density_loss = re.search(r"density error:([\d\.]+)", line)
            if match_val_density_loss:
                val_density_losses[val_epoch_buffer] = float(match_val_density_loss.group(1))


            if (val_epoch_buffer in val_rocon_losses 
                and val_epoch_buffer in val_nc_losses 
                and val_epoch_buffer in val_edge_losses
                and val_epoch_buffer in val_repeat_losses 
                and val_epoch_buffer in val_density_losses 

                ):
                val_epoch_buffer = None

    # return train_losses, val_rocon_losses, val_nc_losses, val_edge_losses
    return train_losses, val_rocon_losses, val_nc_losses, val_edge_losses, val_repeat_losses,  val_density_losses



# def plot_losses(train_losses, val_rocon_losses, val_nc_losses, val_edge_losses, save_path='loss_curve.png'):
def plot_losses(train_losses, val_rocon_losses, val_nc_losses, val_edge_losses, val_repeat_losses,val_density_losses, save_path='loss_curve.png'):

    epochs_train = sorted(train_losses.keys())
    losses_train = [train_losses[e] for e in epochs_train]

    epochs_val = sorted(val_rocon_losses.keys())
    losses_val_rocon = [val_rocon_losses.get(e, None) for e in epochs_val]
    losses_val_nc = [val_nc_losses.get(e, None) for e in epochs_val]
    losses_val_edge = [val_edge_losses.get(e, None) for e in epochs_val]
    losses_val_repeat = [val_repeat_losses.get(e, None) for e in epochs_val]
    losses_val_density = [val_density_losses.get(e, None) for e in epochs_val]

    # 计算组合验证损失：recon + 0.3 × edge + 3 × nc
    losses_val_combined = []
    for e in epochs_val:
        recon = val_rocon_losses.get(e, 0)
        edge = val_edge_losses.get(e, 0)
        nc = val_nc_losses.get(e, 0)
        repeat = val_repeat_losses.get(e, 0)
        density = val_density_losses.get(e, 0)

        combined = recon + 0.3 * edge + 3 * nc + 0.1 * density
        losses_val_combined.append(combined)

    # 绘图
    plt.figure(figsize=(6, 6))
    plt.plot(epochs_train, losses_train, label='Training Loss', color='blue')
    if any(losses_val_rocon):
        plt.plot(epochs_val, losses_val_rocon, label='Validation Recon Loss', color='red', marker='o', linestyle='--')
    if any(losses_val_nc):
        plt.plot(epochs_val, losses_val_nc, label='Validation NC Loss', color='green', marker='^', linestyle='--')
    if any(losses_val_edge):
        plt.plot(epochs_val, losses_val_edge, label='Validation Edge Loss', color='black', marker='s', linestyle='--')
    if any(losses_val_repeat):
        # 直接乘上 0.0001 来进行缩放
        plt.plot(epochs_val, [0.0001 * loss for loss in losses_val_repeat], 
                label='1e-4 * Validation Repeat Loss', color='orange', marker='s', linestyle='--')
    if any(losses_val_density):
        # 直接乘上 0.0001 来进行缩放
        plt.plot(epochs_val, [0.1 * loss for loss in losses_val_density], 
                label='1e-4 * Validation Repeat Loss', color='pink', marker='s', linestyle='--')

    if any(losses_val_combined):
        plt.plot(epochs_val, losses_val_combined, label='Validation Combined Loss', color='purple')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.title('Loss Over Epochs (T1 + T2w + Loss_repeat)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.xlim(0, 150)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved loss plot to: {save_path}")


# 用法
# log_file_path = '/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_monai/log_hemi-0.1left_wm_0003.log'
# save_img_path = '/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_monai/log_hemi-0.1left_wm_0003.png'

# log_file_path = '/root/autodl-tmp/hcp1200/surface/ckpts_all/log_hemi-left_wm_0002.log'
# save_img_path = '/root/autodl-tmp/hcp1200/surface/ckpts_all/log_hemi-left_wm_0002.png'

log_file_path = '/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_repeat_soft_5_1/log_hemi-left_wm_0001.log'
save_img_path = '/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_repeat_soft_5_1/log_hemi-left_wm_0001.png'

# train_losses, val_rocon_losses, val_nc_losses, val_edge_losses  = parse_log_file(log_file_path)
# plot_losses(train_losses, val_rocon_losses, val_nc_losses, val_edge_losses, save_path=save_img_path)
train_losses, val_rocon_losses, val_nc_losses, val_edge_losses, val_repeat_losses, val_density_losses  = parse_log_file(log_file_path)
plot_losses(train_losses, val_rocon_losses, val_nc_losses, val_edge_losses, val_repeat_losses, val_density_losses,save_path=save_img_path)