# 多模态大模型推理基准测试系统
## 环境
python 3.10

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers

pip install -r requirements.txt

## 下载数据集
下载[SEED-Bench-2 · 数据集](https://modelscope.cn/datasets/TencentARC/SEED-Bench-2/summary)
解压缩完需要重命名
cat SEED-Bench-2-image.zip.* > SEED-Bench-2-image.zip
mv cc3m-image/ cc3m/
mv SEED-Bench-2-image/ SEED-Bench-v2/
下载[OpenOrca · 数据集](https://huggingface.co/datasets/Open-Orca/OpenOrca)
数据处理python processorca.py
直接下载处理后的数据集https://gitee.com/delta-1234/processed-open-orca.git


## 下载模型
下载[llava-hf/llava-1.5-7b-hf · Hugging Face](https://huggingface.co/llava-hf/llava-1.5-7b-hf)

## 构建loadgen

```bash
pip install pynvml
pip install requests
pip install pillow
pip install tqdm
pip install datasets
pip install accelerate
pip install pybind11
cd loadgen
CFLAGS="-std=c++14" python setup.py develop
conda install -c conda-forge libstdcxx-ng
```

## 运行

```bash
export CHECKPOINT_PATH="/home/user/buaa/rgzndengtao/model/"
export DATASET_PATH="/home/user/buaa/rgzndengtao/dataset/"

export CHECKPOINT_PATH="/home/buaa/dengtao/model/"
export DATASET_PATH="/home/buaa/dengtao/dataset/"

python -u main.py --scenario Offline --dataset SEED-Bench-2 --model-name llava-1.5-7b-hf --total-sample-count 24576 --batch-size 2 --device cuda --test-mode PerformanceOnly

nohup python -u main.py --scenario Offline --dataset SEED-Bench-2 --model-name llava-1.5-13b-hf --total-sample-count 34576 --batch-size 2 --device cuda --test-mode PerformanceOnly > output.txt 2>&1 &

python -u main.py --scenario Server --dataset SEED-Bench-2 --model-name llava-1.5-7b-hf --total-sample-count 34576 --batch-size 1 --device cuda --test-mode PerformanceOnly

python -u main.py --scenario SingleStream --dataset SEED-Bench-2 --model-name llava-1.5-7b-hf --total-sample-count 34576 --batch-size 1 --device cuda --test-mode PerformanceOnly

python -u main.py --scenario MultiStream --dataset SEED-Bench-2 --model-name llava-1.5-7b-hf --total-sample-count 34576 --batch-size 1 --device cuda --test-mode PerformanceOnly

nohup python -u main.py --scenario MultiStream --dataset SEED-Bench-2 --model-name llava-1.5-13b-hf --total-sample-count 34576 --batch-size 1 --device cuda --test-mode PerformanceOnly > output.txt 2>&1 &
```

```bash
export CHECKPOINT_PATH="/data/dengtao/model/"
export DATASET_PATH="/data/dengtao/dataset/"

python -u main.py \
        --scenario Server \
		--dataset SEED-Bench-2 \
		--model-name llava-1.5-7b \
		--device cuda
```

## 前端与后端

前端https://github.com/delta-1234/diplom_frontend.git

后端https://github.com/delta-1234/diplom_backend.git
