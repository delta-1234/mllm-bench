from model_monitor import ModelMonitor
from gpu_monitor import GPUMonitorProcess
from multiprocessing import Event

# stop_event = Event()

# monitor = GPUMonitorProcess(
#         interval=2,
#         duration=3600,  # 1小时
#         output_file="output-logs/gpu_monitor.jsonl",
#         stop_event=stop_event
#     )

# print(monitor.calculate_gpu_stats())

print(ModelMonitor.calculate_average_times())