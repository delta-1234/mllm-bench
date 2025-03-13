CHECKPOINT_PATH="/root/autodl-tmp/model/hub/"
DATASET_PATH="/root/autodl-tmp/dataset/"

python -u main.py --scenario Offline \
		--dataset mmlu \
		--model-name llama2-7b \
		--total-sample-count 24576 \
		--device cuda
