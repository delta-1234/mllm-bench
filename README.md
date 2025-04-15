# 多模态大模型推理基准测试系统
## 环境
python 3.10
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install -r requirements.txt

## 下载数据集
下载[SEED-Bench-2 · 数据集](https://modelscope.cn/datasets/TencentARC/SEED-Bench-2/summary)

## 下载模型
下载[llava-hf/llava-1.5-7b-hf · Hugging Face](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
下载llava套件https://github.com/haotian-liu/LLaVA
```bash
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## 构建loadgen

```bash

cd loadgen
CFLAGS="-std=c++14" python setup.py develop
conda install -c conda-forge libstdcxx-ng
pip install pynvml
pip install requests
pip install pillow
pip install tqdm
pip install datasets
pip install accelerate
```

## 运行

```bash
export CHECKPOINT_PATH="/data/dengtao/model/"
export DATASET_PATH="/data/dengtao/dataset/"

python -u main.py --scenario MultiStream --dataset SEED-Bench-2 --model-name llava-1.5-7b-hf --total-sample-count 24576 --batch-size 1 --device cuda
nohup python -u main.py --scenario Server --dataset SEED-Bench-2 --model-name llava-1.5-7b-hf --total-sample-count 24576 --batch-size 1 --device cuda > output.txt 2>&1 &
```

手动下载[clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)到~/.cache/huggingface/

设置环境变量强制使用本地模型 export HF_HUB_OFFLINE=1

视觉处理时间: 14.266ms
文本处理时间: 1.225ms
文本生成时间: 2919.674ms
跨模态等待时间: 1.338ms

```bash
export CHECKPOINT_PATH="/data/dengtao/model/"
export DATASET_PATH="/data/dengtao/dataset/"

python -u main.py \
        --scenario Server \
		--dataset SEED-Bench-2 \
		--model-name llava-1.5-7b \
		--device cuda
```