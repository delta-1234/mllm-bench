import time
import json
from typing import ClassVar, Optional, Dict, List, Union
from collections import defaultdict
import os

class ModelMonitor:
    """
    静态监控类，无需实例化，直接通过类名调用方法。
    使用前需先初始化日志文件路径: ModelMonitor.initialize("path/to/log.jsonl")
    """
    _log_file: ClassVar[Optional[str]] = None

    @classmethod
    def initialize(cls, log_file: str) -> None:
        """初始化日志文件路径（必须优先调用）"""
        cls._log_file = log_file
        # 获取文件所在目录
        output_dir = os.path.dirname(log_file)
        
        # 只有当目录非空时才创建目录
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 清空原文件内容
        with open(cls._log_file, "w", encoding="utf-8") as f:
            pass

    @classmethod
    def _record_event(cls, event_type: str, time: int, sample_id: list) -> None:
        """内部方法：记录事件到日志文件"""
        if not cls._log_file:
            raise RuntimeError("ModelMonitor 未初始化，请先调用 initialize() 设置日志路径")

        log_entry = {
            "event": event_type,
            "sample_id": sample_id,  # 新增字段
            "time": time,
        }

        with open(cls._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    @classmethod
    def RecordVisionProcessTime(cls, time: int, sample_id: list) -> None:
        """记录视觉处理时间(ns)"""
        cls._record_event("vision_procrss_time", time, sample_id)

    @classmethod
    def RecordTextProcessTime(cls, time: int, sample_id: list) -> None:
        """记录文本处理时间(ns)"""
        cls._record_event("text_procrss_time", time, sample_id)

    @classmethod
    def RecordTextGenerationTime(cls, time: int, sample_id: list) -> None:
        """记录文本处理时间(ns)"""
        cls._record_event("text_generation_time", time, sample_id)
    
    @classmethod
    def calculate_average_times(cls) -> Dict[str, Union[float, int]]:
        """
        读取日志文件，计算以下指标：
        - 平均视觉处理时间
        - 平均文本生成时间
        - 平均跨模态对齐耗时(vision_procrss_time - text_procrss_time)
        返回结果示例：
        {
            "avg_vision_ms": 65.36, 
            "avg_text_ms": 0.69,
            "avg_align_ms": 1018.82,
            "valid_samples": 1
        }
        """
        if not cls._log_file:
            cls._log_file = "./output-logs/model_monitor.jsonl"  # 默认日志文件路径

        sample_data = {}

        sample_count = 0
        # 读取日志文件（假设路径为类内部配置）
        with open(cls._log_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    event = data["event"]
                    sample_ids = data["sample_id"]  # 可能是多个 ID 的列表
                    time_us = data["time"]
                except (KeyError, json.JSONDecodeError):
                    continue

                # 将事件时间关联到每个独立 sample_id
                if sample_count == 0:
                    sample_count = len(sample_ids)
                for sample_id in sample_ids:
                    sample_str = str(sample_id)  # 统一为字符串标识
                    if sample_str not in sample_data:
                        sample_data[sample_str] = {}
                    sample_data[sample_str][event] = time_us  # 相同时间赋给所有关联样本

        # 统计计算
        valid_samples = 0
        total_vision, total_text_gen, total_align = 0.0, 0.0, 0.0

        for events in sample_data.values():
            # 检查是否包含所有必要事件
            required_events = {"text_procrss_time", "vision_procrss_time", "text_generation_time"}
            if not required_events.issubset(events.keys()):
                continue

            # 时间转换(ns → 微s）
            text_time = events["text_procrss_time"] / sample_count
            vision_time = events["vision_procrss_time"] / sample_count
            text_gen_time = events["text_generation_time"] / sample_count

            # 计算对齐时间
            align_time = vision_time - text_time

            # 累加统计
            total_vision += vision_time
            total_text_gen += text_gen_time
            total_align += align_time
            valid_samples += 1

        # 计算平均值（保留两位小数）
        avg_vision_pr = round(total_vision / valid_samples, 2) if valid_samples else 0
        avg_text_gen = round(total_text_gen / valid_samples, 2) if valid_samples else 0
        avg_align = round(total_align / valid_samples, 2) if valid_samples else 0

        return {
            "avg_vision_ns": avg_vision_pr,
            "avg_text_gen_ns": avg_text_gen,
            "avg_align_ns": avg_align,
            "valid_samples": valid_samples
        }