# 多模态大模型推理基准测试系统
## 环境
python 3.10
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt

## 下载数据集
下载[SEED-Bench-2 · 数据集](https://modelscope.cn/datasets/TencentARC/SEED-Bench-2/summary)
解压缩完需要重命名
mv cc3m-image/ cc3m/
mv SEED-Bench-2-image/ SEED-Bench-v2/
下载[OpenOrca · 数据集](https://huggingface.co/datasets/Open-Orca/OpenOrca)
数据处理python processorca.py


## 下载模型
下载[llava-hf/llava-1.5-7b-hf · Hugging Face](https://huggingface.co/llava-hf/llava-1.5-7b-hf)放在core的openai文件夹下
下载llava套件https://github.com/haotian-liu/LLaVA
```bash
pip install --upgrade pip  # enable PEP 660 support
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install protobuf -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 构建loadgen

```bash

cd loadgen
CFLAGS="-std=c++14" python setup.py develop
conda install -c conda-forge libstdcxx-ng
pip install pynvml -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install requests -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install datasets -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pybind11 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 运行

```bash
export CHECKPOINT_PATH="/home/user/buaa/rgzndengtao/model/"
export DATASET_PATH="/home/user/buaa/rgzndengtao/dataset/"

export CHECKPOINT_PATH="/home/buaa/dengtao/model/"
export DATASET_PATH="/home/buaa/dengtao/dataset/"
CUDA_VISIBLE_DEVICES=1

nohup python -u main.py --scenario SingleStream --dataset SEED-Bench-2 --model-name llava-v1.5-7b --total-sample-count 24576 --batch-size 1 --device cuda --test-mode PerformanceOnly > output.txt 2>&1 &

nohup python -u main.py --scenario Offline --dataset SEED-Bench-2 --model-name llava-v1.5-7b --total-sample-count 34576 --batch-size 1 --device cuda --test-mode AccuracyOnly > output.txt 2>&1 &

python -u main.py --scenario Server --dataset SEED-Bench-2 --model-name llava-1.5-7b-hf --total-sample-count 34576 --batch-size 1 --device cuda --test-mode PerformanceOnly

python -u main.py --scenario SingleStream --dataset SEED-Bench-2 --model-name llava-1.5-7b-hf --total-sample-count 34576 --batch-size 1 --device cuda --test-mode PerformanceOnly

python -u main.py --scenario MultiStream --dataset SEED-Bench-2 --model-name llava-1.5-7b-hf --total-sample-count 34576 --batch-size 1 --device cuda --test-mode PerformanceOnly

nohup python -u main.py --scenario MultiStream --dataset SEED-Bench-2 --model-name llava-1.5-7b-hf --total-sample-count 34576 --batch-size 1 --device cuda --test-mode PerformanceOnly > output.txt 2>&1 &
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