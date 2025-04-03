import json
import matplotlib.pyplot as plt
from datetime import datetime

# 读取并解析数据
gpu0_data = []
with open('monitor.jsonl', 'r') as f:
    for idx, line in enumerate(f):
        # 只取偶数行作为gpu0数据
        if idx % 2 == 0:
            record = json.loads(line)
            gpu0_data.append({
                'timestamp': datetime.fromtimestamp(record['timestamp']),
                'utilization': record['gpu_utilization'],
                'memory_used': record['memory_used'],
                'power': record['power']
            })

# 创建包含三个子图的画布
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 绘制GPU利用率
ax1.plot(
    [d['timestamp'] for d in gpu0_data],
    [d['utilization'] for d in gpu0_data],
    color='tab:blue',
    label='Utilization'
)
ax1.set_ylabel('GPU Utilization (%)')
ax1.grid(True)
ax1.legend()

# 绘制显存占用
ax2.plot(
    [d['timestamp'] for d in gpu0_data],
    [d['memory_used'] for d in gpu0_data],
    color='tab:orange',
    label='Memory Usage'
)
ax2.set_ylabel('VRAM Usage (MB)')
ax2.grid(True)
ax2.legend()

# 绘制功耗曲线
ax3.plot(
    [d['timestamp'] for d in gpu0_data],
    [d['power'] for d in gpu0_data],
    color='tab:red',
    label='Power Draw'
)
ax3.set_ylabel('Power (W)')
ax3.grid(True)
ax3.legend()

# 设置公共参数
plt.xlabel('Time')
plt.suptitle('GPU0 Monitoring Metrics')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 在 plt.show() 之前添加保存代码
plt.tight_layout()
plt.savefig(
    'gpu0_monitoring.png',  # 输出文件名
    dpi=300,               # 分辨率设置（推荐300dpi以上）
    bbox_inches='tight'    # 自动裁剪白边
)
plt.show()