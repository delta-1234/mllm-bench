import os
import json
import argparse
import torch
from tqdm import tqdm
import numpy as np
import random
import pdb
import re
from PIL import Image

# python llava_eval.py --anno_path ~/SEED-Bench-2/SEED-Bench_v2_level1_2_3.json 

# root directory of cc3m
cc3m_dir = "/home/dt/SEED-Bench-2/cc3m-image"
# root directory of seed bench v2
seed_bench_v2_dir = "/home/dt/SEED-Bench-2/seed_bench_image_v2"
# 选择第二张 GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 模型路径
MODEL_PATH = "/home/dt/llava-1.5-7b-hf"

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def filter_questions(data, level='L2', subpart='all', version='v2'):
    if level == "L1":
        valid_level_data = ['L1']
    elif level == "L2":
        valid_level_data = ['L1', 'L2']
    elif level == "L3":
        valid_level_data = ['L1', 'L2', 'L3']
    else:
        raise ValueError(f"Invalid level: {level}")
    data = [q for q in data if q["level"] in valid_level_data]

    if subpart in ['Single-Image & Text Comprehension', 'Multiple-Images & Text Comprehension', 'Video & Text Comprehension', 'Interleaved Image & Text Comprehension', 'Image Generation', 'Image & Text Generation']:
        valid_subgroup_data = subpart
    elif subpart == 'all':
        valid_subgroup_data = ['Single-Image & Text Comprehension', 'Multiple-Images & Text Comprehension', 'Video & Text Comprehension', 'Interleaved Image & Text Comprehension', 'Image Generation', 'Image & Text Generation']
    else:
        raise ValueError(f"Invalid subpart: {subpart}")
    data = [q for q in data if q["subpart"] in valid_subgroup_data]

    if version == 'v1':
        valid_version_data = ['v1']
    elif version == 'v2':
        valid_version_data = ['v1', 'v2']
    else:
        raise ValueError(f"Invalid version: {version}")
    data = [q for q in data if q["version"] in valid_version_data]

    return data

def build_model(model_name):
    if model_name == 'llava_1.5':
        # 直接加载本地的llava模型和tokenizer
        # 加载模型和 processor
        from transformers import LlavaForConditionalGeneration, LlavaProcessor
        processor = LlavaProcessor.from_pretrained(MODEL_PATH, use_fast=True)
        model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
        return model, processor
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def is_integer_string(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def run_inference(model, processor, qa_anno, output_dir):
    total_qa_num = len(qa_anno)
    answer_list = []
    output_f = open(os.path.join(output_dir, "results.json"), "a")
    step = 0
    correct_num = 0
    all_num = 0
    for qa_item in tqdm(qa_anno):
        if qa_item["data_source"] == 'cc3m':
            question = '<image>\nquestion:' + qa_item['question'] + 'choices:' + 'A.' + qa_item['choice_a'] + ' B.' + \
                       qa_item['choice_b'] + ' C.' + qa_item['choice_c'] + ' D.' + qa_item['choice_d'] + '\n'
            image_dir = cc3m_dir
        elif qa_item["data_source"] == 'SEED-Bench v2':
            image_dir = seed_bench_v2_dir
            # TODO 没有空间解压了先跳过
            continue
        else:
            raise ValueError("The data type is not valid.")

        try:
            image_path = image_dir + '/' + qa_item['data_id']
            image = Image.open(image_path).convert("RGB")
            # 将图像和问题转换为模型可以理解的输入
            inputs = processor(images=image, text=question, return_tensors="pt", do_resize=True).to(device)

            # 生成回答
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=128)

            # 解码输出
            response = processor.batch_decode(output, skip_special_tokens=True)[0]
            # print(image_path)
            # print(response)
            # 从生成的答案中选择正确的选项
            response = response[len(question)-1:]
            match = re.search(r'[ABCD]', response)
            pred_id = match.group()
            
            # 记录答案
            answer_record = {
                'question_id': qa_item['question_id'],
                'correct_answer': qa_item['answer'],
                'prediction': pred_id
            }
            all_num += 1
            if pred_id == qa_item['answer']:
                correct_num += 1
            answer_list.append(answer_record)
            # output prediction record for each question
            output_f.write(json.dumps(answer_record) + "\n")
        except:
            pass
        step += 1
    print('acc:' + str(correct_num/all_num))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--model', type=str, default='llava_1.5')  # 本地llava模型
    parser.add_argument('--anno_path', type=str, default='SEED-Bench_v2_level1_2_3.json')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--evaluate_level', type=str, default='L2')
    parser.add_argument('--evaluate_part', type=str, default='all')
    parser.add_argument('--evaluate_version', type=str, default='v2')
    args = parser.parse_args()

    qa_anno = json.load(open(args.anno_path, 'rb'))
    if 'questions' in qa_anno.keys():
        qa_anno = qa_anno['questions']
    qa_anno = filter_questions(qa_anno, args.evaluate_level, args.evaluate_part, args.evaluate_version)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print(f'evaluating.. {args.model}')
    # 直接加载模型和tokenizer
    model, processor = build_model(args.model)

    run_inference(model, processor, qa_anno, args.output_dir)
