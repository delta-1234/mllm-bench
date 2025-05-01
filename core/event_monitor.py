class EventMonitor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.results = {
            'summary': {
                'scenario': None,
                'samples_per_sec': None,
                'tokens_per_sec': None
            },
            'additional_stats': {}
        }

    def _convert_value(self, value_str):
        """智能转换数值类型"""
        try:
            if '.' in value_str:
                return float(value_str)
            return int(value_str.replace(',', ''))  # 处理千分位分隔符
        except ValueError:
            return value_str.strip()

    def read_results(self):
        with open(self.file_path, 'r') as f:
            section = None
            current_key = None

            for line in f:
                line = line.strip()
                if not line or '===' in line:
                    continue

                # 识别章节
                if line == "MLPerf Results Summary":
                    section = "summary"
                    continue
                elif line == "Additional Stats":
                    section = "stats"
                    continue

                # 解析主要内容
                if section == "summary":
                    if ':' in line:
                        key, value = map(str.strip, line.split(':', 1))
                        
                        # 捕获关键指标
                        if key == "Scenario":
                            self.results['summary']['scenario'] = value
                        elif key == "Samples per second":
                            self.results['summary']['samples_per_sec'] = self._convert_value(value)
                        elif key == "Tokens per second":
                            self.results['summary']['tokens_per_sec'] = self._convert_value(value)
                        elif key == "Completed samples per second":
                            self.results['summary']['completed_samples_per_sec'] = self._convert_value(value)
                        elif key == "Completed tokens per second":
                            self.results['summary']['completed_tokens_per_sec'] = self._convert_value(value)
                        elif key == "90.0th percentile latency (ns)":
                            self.results['summary']['90_per_latency_ns'] = self._convert_value(value)
                        elif key == "99.0th percentile latency (ns)":
                            self.results['summary']['99_per_latency_ns'] = self._convert_value(value)

                elif section == "stats":
                    if ':' in line:
                        key, value = map(str.strip, line.split(':', 1))
                        self.results['additional_stats'][key] = self._convert_value(value)

        return self.results

# 使用示例
if __name__ == "__main__":
    monitor = EventMonitor("./output-logs-Server/mlperf_log_summary.txt")
    data = monitor.read_results()
    
    print(f"测试场景: {data['summary']['scenario']}")
    # print(data)
    print(f"样本吞吐量: {data['summary']['completed_samples_per_sec']:.2f} samples/sec")
    print(f"Token吞吐量: {data['summary']['completed_tokens_per_sec']:,.0f} tokens/sec\n")
    
    print("详细统计数据:")
    for stat, value in data['additional_stats'].items():
        if isinstance(value, float):
            print(f"{stat}: {value:.2f}")
        else:
            print(f"{stat}: {value:,}")  # 格式化大数字显示
