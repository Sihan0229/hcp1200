'''
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'
# 读取日志
with open("/root/autodl-tmp/hcp1200/preprocess_all.log", "r") as f:
    lines = f.readlines()

# 初始化
times = []
dice_values = []
vertices_before = []
vertices_after = []

# 正则表达式匹配
time_pattern = re.compile(r"\[(\d+):(\d+)")
dice_pattern = re.compile(r"Dice after registration:\s*([0-9.]+)")
vertex_pattern = re.compile(r"num of vertices before/after:\s*(\d+)\s*/\s*(\d+)")

# 遍历解析
for line in lines:
    # 处理时间
    time_match = time_pattern.search(line)
    if time_match:
        minutes = int(time_match.group(1))
        seconds = int(time_match.group(2))
        total_seconds = minutes * 60 + seconds
        times.append(total_seconds)

    # Dice
    dice_match = dice_pattern.search(line)
    if dice_match:
        dice_values.append(float(dice_match.group(1)))

    # 顶点数
    vertex_match = vertex_pattern.search(line)
    if vertex_match:
        vertices_before.append(int(vertex_match.group(1)))
        vertices_after.append(int(vertex_match.group(2)))

# 输出统计信息
print("==== 处理时间统计 ====")
print(f"均值: {np.mean(times):.2f} s")
print(f"方差: {np.var(times):.2f}")
print(f"最大值: {np.max(times)} s")
print(f"最小值: {np.min(times)} s")


print("\n==== Dice 值统计 ====")
print(f"均值: {np.mean(dice_values):.4f}")
print(f"方差: {np.var(dice_values):.6f}")
print(f"最大值: {np.max(dice_values):.4f}")
print(f"最小值: {np.min(dice_values):.4f}")


# 绘图函数
import seaborn as sns

def plot_kde_distribution(data, title, xlabel, filename):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data, fill=True, color='skyblue')
    # plt.xlim(100000, 200000) 
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# 绘制三个分布图
# plot_kde_distribution(vertices_before, "Before", "vol count", "vertices_before.png")
# plot_kde_distribution(vertices_after, "After", "vol count", "vertices_after.png")
plot_kde_distribution(dice_values, "Dice", "Dice value", "dice_distribution.png")
'''

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 你的日志文本，建议你用 open() 读入文件也可以
with open("/root/autodl-tmp/hcp1200/preprocess_all.log", "r", encoding="utf-8") as f:
    log_text = f.read()

# 提取 [MM:SS] 时间戳
time_matches = re.findall(r"\[(\d{2}):(\d{2})", log_text)
time_seconds = [int(m)*60 + int(s) for m, s in time_matches]

# 计算每个样本的处理时间（差值）
processing_times = [t2 - t1 for t1, t2 in zip(time_seconds[:-1], time_seconds[1:])]

# 输出统计量
mean_time = np.mean(processing_times)
std_time = np.std(processing_times)
max_time = np.max(processing_times)
min_time = np.min(processing_times)


print(f"样本处理时间统计：")
print(f"均值：{mean_time:.2f} 秒")
print(f"方差：{std_time:.2f} 秒")
print(f"最大值：{max_time:.2f} 秒")
print(f"最小值：{min_time:.2f} 秒")

