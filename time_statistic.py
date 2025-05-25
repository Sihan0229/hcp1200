from datetime import datetime
import numpy as np

log_path = '/root/autodl-tmp/hcp1200/surface/ckpts_all_multi_repeat_5_1/log_hemi_soft_repeat-left_wm_0001.log'  # 替换为你的日志文件路径


epoch_times = []

with open(log_path, 'r') as f:
    for line in f:
        if 'epoch:' in line and 'loss:' in line:
            parts = line.split()
            timestamp_str = f"{parts[0]} {parts[1].split(',')[0]}"
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            epoch_times.append(timestamp)

# 计算每个 epoch 的训练耗时（分钟）
durations = []
for i in range(1, len(epoch_times)):
    duration = (epoch_times[i] - epoch_times[i - 1]).total_seconds() / 60.0
    durations.append(duration)

# 打印每个 epoch 的训练时间
# print("Epoch\tStart Time\t\t\tDuration (min)")
# for i in range(len(durations)):
#     print(f"{i:5d}\t{epoch_times[i]}\t{durations[i]:.2f}")

# 统计分析
durations_np = np.array(durations)
print("\n--- Training Time Statistics (minutes) ---")
print(f"Average: {durations_np.mean():.2f}")
print(f"Max    : {durations_np.max():.2f}")
print(f"Min    : {durations_np.min():.2f}")
print(f"Std Dev: {durations_np.std():.2f}")
print(f"Var    : {durations_np.var():.2f}")

