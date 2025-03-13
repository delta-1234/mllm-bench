# 多模态大模型推理基准测试系统
## 下载数据集
下载[SEED-Bench-2 · 数据集](https://modelscope.cn/datasets/TencentARC/SEED-Bench-2/summary)

## 下载模型
下载[llava-hf/llava-1.5-7b-hf · Hugging Face](https://huggingface.co/llava-hf/llava-1.5-7b-hf)

## 构建loadgen

```bash
cd loadgen
CFLAGS="-std=c++14" python setup.py develop
```

## 运行

```bash
export CHECKPOINT_PATH="/root/xxx/model/"
export DATASET_PATH="/root/xxx/dataset/"

python -u main.py --scenario Offline \
		--dataset mmlu or --dataset /root/xxxx/mmbench/examples/example_dataset.json
		--model-name llama2-7b \
		--total-sample-count 24576 \
		--device cuda
```