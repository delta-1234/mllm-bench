import time
import pynvml
import subprocess
import json
from multiprocessing import Process, Event
import os
from collections import defaultdict
from typing import Dict, List, Tuple

class GPUMonitorProcess(Process):
    def __init__(self, interval=2, duration=60, output_file="gpu_stats.jsonl", stop_event=None):
        super().__init__()
        self.interval = interval
        self.duration = duration
        self.output_file = output_file
        self.stop_event = stop_event or Event()
        self._file_handle = None

    def _get_gpu_power(self):
        """通过nvidia-smi获取GPU功耗数据"""
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            return [round(float(x.strip()), 2) for x in output.strip().split("\n")]
        except Exception as e:
            print(f"获取功耗失败: {str(e)}")
            return None

    def _get_gpu_info(self):
        """获取GPU详细信息"""
        pynvml.nvmlInit()
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            gpu_info_list = []

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # 获取基础信息
                name = pynvml.nvmlDeviceGetName(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                # 转换为MB单位
                gpu_info = {
                    "gpu_id": f"cuda:{i}",
                    "gpu_name": name.decode("utf-8") if isinstance(name, bytes) else name,
                    "memory_used": round(memory_info.used / (1024**2)),  # MB
                    "memory_total": round(memory_info.total / (1024**2)), # MB
                    "gpu_utilization": utilization.gpu,                   # %
                    "temperature": temperature                           # ℃
                }
                gpu_info_list.append(gpu_info)

            return gpu_info_list
        finally:
            pynvml.nvmlShutdown()

    def _init_file(self):
        """初始化输出文件"""
        # 获取文件所在目录
        output_dir = os.path.dirname(self.output_file)
        
        # 只有当目录非空时才创建目录
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 仅清空文件，不写入内容
        with open(self.output_file, "w", encoding="utf-8") as f:
            pass
        # 打开文件（自动创建新文件）
        self._file_handle = open(self.output_file, "a", encoding="utf-8")

    def _write_data(self, timestamp, gpu_data, power_data):
        """写入单条GPU记录"""
        for idx, gpu in enumerate(gpu_data):
            # 合并功耗数据
            record = {
                "timestamp": timestamp,
                **gpu,
                "power": power_data[idx] if power_data and idx < len(power_data) else None
            }
            # 写入JSON行
            self._file_handle.write(json.dumps(record) + "\n")
            self._file_handle.flush()  # 确保立即写入

    def run(self):
        """监控主循环"""
        self._init_file()
        start_time = time.time()
        
        try:
            while not self.stop_event.is_set():
                cycle_start = time.time()
                try:
                    # 获取当前时间戳
                    timestamp = round(time.time(), 2)  # 保留两位小数
                    
                    # 获取数据
                    gpu_info = self._get_gpu_info()
                    power_data = self._get_gpu_power()
                    
                    # 写入文件
                    self._write_data(timestamp, gpu_info, power_data)

                    # 检查持续时间
                    if self.duration and (time.time() - start_time) > self.duration:
                        break

                    # 精确间隔控制
                    elapsed = time.time() - cycle_start
                    sleep_time = max(0, self.interval - elapsed)
                    time.sleep(sleep_time)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"监控进程异常: {str(e)}")
                    break
        finally:
            if self._file_handle:
                self._file_handle.close()

    @classmethod
    def calculate_gpu_stats(self) -> Dict[str, Dict]:
        """
        计算每张显卡的平均利用率(%)、平均显存占用(MB)和总能量消耗(J)
        """
        # 数据结构: {gpu_id: [(util, mem, power, time), ...]}
        gpu_records = defaultdict(list)
        
        output_file = "./output-logs/gpu_monitor.jsonl"  # 默认日志文件路径
        # 阶段1: 读取并预处理数据
        with open(output_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    # 验证必要字段存在
                    required_fields = ["gpu_id", "gpu_utilization", 
                                    "memory_used", "power", "timestamp"]
                    for field in required_fields:
                        if field not in record:
                            raise ValueError(f"Missing field: {field}")
                    
                    # 收集有效数据
                    gpu_id = record["gpu_id"]
                    gpu_records[gpu_id].append((
                        float(record["gpu_utilization"]),
                        int(record["memory_used"]),
                        float(record["power"]),
                        float(record["timestamp"])
                    ))
                    
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    print(f"忽略第 {line_num} 行数据错误: {str(e)}")
                    continue

        # 阶段2: 计算统计指标
        result = {}
        for gpu_id, records in gpu_records.items():
            if len(records) < 2:
                print(f"警告: {gpu_id} 有效数据不足，至少需要2条记录")
                continue
                
            # 按时间排序
            sorted_records = sorted(records, key=lambda x: x[3])
            
            # 初始化统计变量
            total_util = 0.0
            total_mem = 0
            total_energy_j = 0.0  # 能量（焦耳）
            
            # 遍历计算
            prev_time = sorted_records[0][3]
            for util, mem, power, time in sorted_records:
                # 累加利用率/显存（最后会取平均）
                total_util += util
                total_mem += mem
                
                # 计算时间差（单位：秒）
                time_delta = time - prev_time
                if time_delta <= 0:
                    continue  # 忽略无效时间差
                    
                # 能量 = 功率（瓦特） × 时间（秒） → 焦耳
                total_energy_j += power * time_delta
                prev_time = time
                
            # 计算平均值
            num_samples = len(sorted_records)
            avg_util = total_util / num_samples
            avg_mem = total_mem // num_samples  # 取整数MB
            
            result[gpu_id] = {
                "avg_utilization": round(avg_util, 1),
                "avg_memory_used_mb": avg_mem,
                "total_energy_j": round(total_energy_j, 2)
            }
            
        return result

if __name__ == "__main__":
    monitor = GPUMonitorProcess(
        interval=2,
        duration=60,
        output_file="gpu_stats.json"
    )
    monitor.start()
    monitor.join()