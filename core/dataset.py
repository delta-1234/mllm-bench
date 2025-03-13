import os
import time
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from typing import Optional, Dict, Sequence
import io
import copy
import json

import sys
from pathlib import Path
from pprint import pprint

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


from dataload import DatasetLoader,SUPPORTED_DATASETS, SUPPORTED_DATASETS_VLM

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class Datasets:
    def __init__(
        self,
        model_name,
        dataset,
        batch_size=1,
        pad_val=1,
        pad_max=196,
        total_sample_count=24576,
        perf_count_override=None,
        device="cuda",
        tokenizer=None
    ):
        print("Constructing QSL")

        self.model_path = os.getenv('CHECKPOINT_PATH',default = None)

        self.dataset = dataset
        self.model_name = model_name
        self.batch_size = batch_size
        self.pad_val = pad_val
        self.pad_max = pad_max
        self.device=device

        # Check if self.dataset is a JSON file
        if isinstance(self.dataset, str) and self.dataset.endswith('.json'):
            print("Loading JSON dataset file")
            self.list_data_dict = jload(self.dataset)
        else:
            print("Dataset is not a JSON file")
            self.mydataset = DatasetLoader.load_dataset(self.dataset)
            self.list_data_dict = self.mydataset.data

        print(self.model_path + self.model_name)
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path + self.model_name,
                model_max_length=2048,
                padding_side="left",
                use_fast=False,)
        else:
            self.tokenizer = tokenizer

        self.tokenizer.pad_token = self.tokenizer.eos_token


        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        self.sources = []
        for example in self.list_data_dict:
            # 检查是否存在 instruction 字段
            if "instruction" not in example:
                # 如果没有 instruction，将 input 作为 instruction
                example["instruction"] = example.get("input", "")
                example["input"] = ""

            # 根据是否有 input 选择不同的 prompt 模板
            if example.get("input", "").strip():
                prompt = prompt_input.format_map(example)
            else:
                prompt = prompt_no_input.format_map(example)
            self.sources.append(prompt)


        self.targets = [
            f"{example.get('target') or example.get('output', '')}" for example in self.list_data_dict]

        self.source_encoded_input_ids, self.source_encoded_attn_masks, self.input_lens = self.encode_samples()

        max_samples=min(total_sample_count, len(self.sources))

        self.count = max_samples
        self.total_sample_count = max_samples
        self.perf_count = self.count

        self.input_ids = self.source_encoded_input_ids
        self.attention_masks = self.source_encoded_attn_masks


    def encode_samples(self):

        total_samples = len(self.sources)

        source_encoded_input_ids = []
        source_encoded_attn_masks = []
        source_encoded_input_lens = []

        for i in range(total_samples):
            source_encoded = self.tokenizer(self.sources[i], return_tensors="pt",
                padding=True, truncation=True,max_length=1919)
            source_encoded_input_ids.append(source_encoded.input_ids.to(self.device))
            source_encoded_attn_masks.append(source_encoded.attention_mask)
            source_encoded_input_lens.append(source_encoded.input_ids.shape[-1])

        return source_encoded_input_ids, source_encoded_attn_masks, source_encoded_input_lens

    def postProcess(self, out_tokens, input_seq_lens=None, query_id_list=None, sample_index_list=None):

        # Everything is padded to max_len (1024), so prune the input and parse to numpy
        output_seq = out_tokens[:, 1024:].cpu().numpy()
        assert len(query_id_list) == output_seq.shape[0]

        return output_seq

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def __del__(self):
        print("Finished destroying QSL.")
