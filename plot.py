import matplotlib.pyplot as plt
import numpy as np
# 读取log.txt文件内容
with open('/data/ljf/3DShape2VecSet/output_coarse_8192/dm/kl_d512_m512_l8_d24_edm/log.txt', 'r') as f:
    log_content = f.readlines()

# 提取train_loss值
train_losses = [float(line.split('"train_loss": ')[1].split(', ')[0]) for line in log_content if '"train_loss":' in line]

# 设置图表大小
plt.figure(figsize=(10, 6))

# 绘制train_loss随epoch的变化图
epochs = range(len(train_losses))
epochs = np.array(epochs)[::5]
train_losses = np.array(train_losses)[::5]
plt.plot(epochs, train_losses, linestyle='-', color='b')  # 设置线型和颜色

# 设置轴标签和标题
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Train Loss', fontsize=14)
plt.title('Train Loss per Epoch', fontsize=16)

# 显示网格
# plt.grid(True)

# 确保轴标签显示
# plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# 自动调整子图参数，使之填充整个图表区域
# plt.tight_layout()

# 保存图表
plt.savefig('./plot.png')

# 显示图表
# plt.show()
