from model_monitor import ModelMonitor
from gpu_monitor import GPUMonitorProcess
from event_monitor import EventMonitor

result = ModelMonitor.calculate_average_times()
result.update(GPUMonitorProcess.calculate_gpu_stats())
result.update(EventMonitor.read_results())
print(result)


