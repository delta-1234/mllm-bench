from model_monitor import ModelMonitor
from gpu_monitor import GPUMonitorProcess

result = ModelMonitor.calculate_average_times()
result.update(GPUMonitorProcess.calculate_gpu_stats())
print(result)

