import re
from random import sample

from model_monitor import ModelMonitor
from gpu_monitor import GPUMonitorProcess
from event_monitor import EventMonitor
import requests

import json


def calculate_accuracy(file_path):
    total = 0
    correct = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data["correct_answer"] == data["prediction"]:
                    correct += 1
                total += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"跳过无效行（错误：{e}）: {line}")

    if total == 0:
        return 0.0

    accuracy = correct / total * 100
    return round(accuracy, 2)

# 输出示例 → 准确率: 72.34% (1234/1705)

result = ModelMonitor.calculate_average_times()
result.update(GPUMonitorProcess.calculate_gpu_stats())
monitor = EventMonitor("./output-logs/mlperf_log_summary.txt")
result.update(monitor.read_results())

result.update({'accuracy': calculate_accuracy("results/results.json")})

model_name = ''
sample_num = 0
with open("./output-logs/output.txt", 'r', encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if i == 5:  # 第6行对应索引5
            model_name = line.split(':', 1)[1].strip()  # 取冒号后的内容并去空格
        match = re.search(r'Samples run:\s*(\d+)', line)
        if match:
            sample_num = int(match.group(1))

result.update({'sample_number': sample_num})
result.update({'model_name': model_name})

result.update({'user_name': "delta"})
result.update({'password': "88888888"})

print(result)
url = "http://127.0.0.1:8000/api/upload_data"

try:
    # 发送POST请求，json参数会自动设置Content-Type为application/json
    response = requests.post(url, json=result, timeout=5)

    # 检查响应状态码
    if response.status_code == 200:
        print("数据上传成功！")
        # 如果接口返回JSON数据，可以解析响应内容
        print("响应内容:", response.json())
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print("错误信息:", response.text)

except requests.exceptions.RequestException as e:
    # 处理请求异常（如网络错误、超时等）
    print(f"请求发生异常: {e}")
except ValueError as ve:
    # 处理JSON解析错误（如果响应不是合法的JSON）
    print(f"解析响应JSON失败: {ve}")


