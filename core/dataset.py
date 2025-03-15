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


from dataload import DatasetLoader

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
        dataset_name,
        batch_size=1,
        pad_val=1,
        pad_max=196,
        total_sample_count=24576,
        device="cuda",
        processor=None
    ):
        print("Constructing QSL")

        self.model_path = os.getenv('CHECKPOINT_PATH',default = None)

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.pad_val = pad_val
        self.pad_max = pad_max
        self.device=device

        print("Loading dataset file")
        self.dataset = DatasetLoader.load_dataset(self.dataset_name)
        self.list_data_dict = self.dataset.data

        # print(self.model_path + self.model_name)
        if processor is None:
            print("processor is None")
            sys.exit(1)
        else:
            self.processor = processor

        self.source_encoded_input_ids, self.targets = self.encode_samples()

        max_samples=min(total_sample_count, len(self.targets))

        self.count = max_samples
        self.total_sample_count = max_samples
        self.perf_count = self.count
        self.input_ids = self.source_encoded_input_ids

    def encode_samples(self):
        source_encoded_input_ids = []
        target = []
        for example in self.dataset:
            source_encoded = self.processor(text=example["question"], images=example['images'],
                                            return_tensors="pt", padding=True)
            source_encoded_input_ids.append(source_encoded)
            target.append(example["answer"])
        return source_encoded_input_ids, target

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
