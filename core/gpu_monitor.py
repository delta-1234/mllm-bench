import time
import pynvml
import subprocess
import json
from multiprocessing import Process, Event
import os

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

if __name__ == "__main__":
    monitor = GPUMonitorProcess(
        interval=2,
        duration=60,
        output_file="gpu_stats.json"
    )
    monitor.start()
    monitor.join()