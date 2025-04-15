import json
# import matplotlib.pyplot as plt
from datetime import datetime

# 读取并解析数据
gpu0_data = []
with open('monitor.jsonl', 'r') as f:
    for idx, line in enumerate(f):
        if idx % 2 == 0:  # 只取偶数行为 GPU0
            record = json.loads(line)
            gpu0_data.append({
                'timestamp': datetime.fromtimestamp(record['timestamp']),
                'timestamp_raw': record['timestamp'],  # 保留原始时间戳用于计算能量
                'utilization': record['gpu_utilization'],
                'memory_used': record['memory_used'],
                'power': record['power']
            })

# 计算平均值
avg_util = sum(d['utilization'] for d in gpu0_data) / len(gpu0_data)
avg_mem = sum(d['memory_used'] for d in gpu0_data) / len(gpu0_data)

# 计算总能量（Joule）
total_energy = 0.0
for i in range(1, len(gpu0_data)):
    power = gpu0_data[i]['power']  # 以后一项的功率为准（也可用平均）
    duration = gpu0_data[i]['timestamp_raw'] - gpu0_data[i - 1]['timestamp_raw']
    total_energy += power * duration  # 单位：瓦特 * 秒 = 焦耳

print(f"平均 GPU 利用率: {avg_util:.2f}%")
print(f"平均显存占用: {avg_mem:.2f} MB")
print(f"总能量消耗: {total_energy:.2f} J")

# # 创建包含三个子图的画布
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# # 绘制GPU利用率
# ax1.plot(
#     [d['timestamp'] for d in gpu0_data],
#     [d['utilization'] for d in gpu0_data],
#     color='tab:blue',
#     label='Utilization'
# )
# ax1.set_ylabel('GPU Utilization (%)')
# ax1.grid(True)
# ax1.legend()

# # 绘制显存占用
# ax2.plot(
#     [d['timestamp'] for d in gpu0_data],
#     [d['memory_used'] for d in gpu0_data],
#     color='tab:orange',
#     label='Memory Usage'
# )
# ax2.set_ylabel('VRAM Usage (MB)')
# ax2.grid(True)
# ax2.legend()

# # 绘制功耗曲线
# ax3.plot(
#     [d['timestamp'] for d in gpu0_data],
#     [d['power'] for d in gpu0_data],
#     color='tab:red',
#     label='Power Draw'
# )
# ax3.set_ylabel('Power (W)')
# ax3.grid(True)
# ax3.legend()

# # 设置公共参数
# plt.xlabel('Time')
# plt.suptitle('GPU0 Monitoring Metrics')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # 在 plt.show() 之前添加保存代码
# plt.tight_layout()
# plt.savefig(
#     'gpu0_monitoring.png',  # 输出文件名
#     dpi=300,               # 分辨率设置（推荐300dpi以上）
#     bbox_inches='tight'    # 自动裁剪白边
# )
# plt.show()