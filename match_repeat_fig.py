'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_match_count_histogram(data, title, xlabel, filename):
    plt.figure(figsize=(8, 5))
    sns.histplot(data, bins=range(data.min(), data.max() + 2), color='skyblue', edgecolor='black', stat="count")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of GT points")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 加载数据
counts = np.load("match_counts.npy")

# 可选：只看被匹配过的点（如果有大量为0的点）
# counts = counts[counts > 0]

# 绘图并保存
plot_match_count_histogram(
    data=counts,
    title="GT Point Match Count Histogram",
    xlabel="Times a GT point was matched",
    filename="match_count_histogram.png"
)
'''
import numpy as np

# 加载 match_counts.npy
counts = np.load("match_counts.npy")

# 使用 numpy 统计唯一值和它们出现的次数
unique, freqs = np.unique(counts, return_counts=True)

# 打印结果
print("匹配次数\tGT 点数量")
for u, f in zip(unique, freqs):
    print(f"{int(u)} 次\t\t{int(f)} 个")
