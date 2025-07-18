from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import numpy as np
import array

import requests
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import mlperf_loadgen as lg
import pickle
import time
import threading
import tqdm
import queue
from pathlib import Path

from model_monitor import ModelMonitor
from modeload import ModelLoader, SUPPORTED_MODELS, SUPPORTED_MODELS_VLM
from dataset import Datasets
from transformers.generation.streamers import BaseStreamer

gen_kwargs = {
    "early_stopping": False,
    "max_new_tokens": 1024,
    "min_new_tokens": 1,
    "num_beams": 1,
    "do_sample": False,
}

class SUT_native_base:
    def __init__(
        self,
        model_name=None,
        dtype="bfloat16",
        batch_size=None,
        dataset_name=None,
        total_sample_count=24576,
        device="cuda",
        workers=1,
        test_mode=lg.TestMode.PerformanceOnly,
        use_cached_outputs=False,
    ):
        # load dataset
        self.tokenizer = None
        self.processor = None
        self.model_name = model_name

        self.device = device

        if not batch_size:
            if device == "cpu":
                batch_size = 1
            else:
                batch_size = 4  # Reduce to 8 if using 4 GPUs, 16 for 8.
        self.batch_size = batch_size

        # dtype
        if dtype == "bfloat16":
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
        elif dtype == "float16":
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32

        if "cuda" in self.device:
            assert torch.cuda.is_available(), "torch gpu is not available, exiting..."

        self.dataset_name = dataset_name
        print(f"Dataset Name: {self.dataset_name}")
        print(f"Model Name:{self.model_name}")

        self.load_model()

        self.data_object = Datasets(
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            batch_size=self.batch_size,
            total_sample_count=total_sample_count,
            device=self.device,
            processor=self.processor,
        )

        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )


        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()

        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

        self.test_mode = test_mode

    def start(self):
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

    def process_queries(self):
        """排队查询的处理器。用户可以选择添加批处理逻辑"""
        if self.test_mode == lg.TestMode.AccuracyOnly:
            self.process_queries_acc()
            return
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            query_ids = [q.index for q in qitem]

            # 生成缓存文件名
            fname = "q" + "_".join([str(i) for i in query_ids])
            fname = f"run_outputs/{fname}.pkl"
            _p = Path(fname)

            # 如果开启缓存，并且文件已存在，则读取缓存
            if self.use_cached_outputs and _p.exists():
                # Read cache
                with _p.open(mode="rb") as f:
                    d = pickle.load(f)
                processed_output = d["outputs"]
                tik1 = tik2 = tik3 = tok = None
            else:
                tik1 = time.time()
                questions = []
                images_list = []
                sample_ids = []
                for q in qitem:
                    sample = self.data_object.dataset[q.index]
                    questions.append(sample["question"])
                    images_list.extend(sample["images"])
                    sample_ids.append(sample["question_id"])
                
                if len(images_list) == 0:
                    images_list = None
                # 批输入,原本是tokenizier
                inputs = self.processor(
                    text=questions,
                    images=images_list, # 视频关键帧
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                tik2 = time.time()

                # 生成模型输出
                pred_output_tokens = self.model.generate(**inputs)

                tik3 = time.time()

                # 处理模型输出
                processed_output = self.processor.batch_decode(
                    pred_output_tokens,
                    skip_special_tokens=True
                )
                # print(processed_output)

            # 发送 LoadGen 响应
            for i in range(len(qitem)):
                output = processed_output[i]
                n_tokens = len(output)
                response_array = array.array("B", output.encode("utf-8"))
                bi = response_array.buffer_info()
                response = [lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)]
                lg.QuerySamplesComplete(response)

            tok = time.time()

            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                print(f"Samples run: {self.sample_counter}")
                if tik1:
                    print(f"\tBatchMaker time: {tik2 - tik1}")
                    print(f"\tInference time: {tik3 - tik2}")
                    print(f"\tPostprocess time: {tok - tik3}")
                    print(f"\t==== Total time: {tok - tik1}")
                else:
                    print(f"\tLoaded from cache: {_p}")
    
    def process_queries_acc(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            questions = []
            images_list = []
            sample_ids = []
            for q in qitem:
                sample = self.data_object.dataset[q.index]
                questions.append(sample["question"])
                images_list.extend(sample["images"])
                sample_ids.append(sample["question_id"])
            
            if len(images_list) == 0:
                images_list = None
            # 批输入,原本是tokenizier
            tik1 = time.time_ns()
            inputs = self.processor(
                text=questions,
                images=None, # 无图片
                return_tensors="pt",
                padding=True
            ).to(self.device)
            tik2 = time.time_ns()
            ModelMonitor.RecordTextProcessTime(tik2-tik1, sample_ids)

            tik1 = time.time_ns()
            inputs = self.processor(
                text=questions,
                images=images_list, # 视频关键帧
                return_tensors="pt",
                padding=True
            ).to(self.device)
            tik2 = time.time_ns()
            ModelMonitor.RecordVisionProcessTime(tik2-tik1, sample_ids)

            # 生成模型输出
            tik1 = time.time_ns()
            pred_output_tokens = self.model.generate(**inputs)

            # 处理模型输出
            processed_output = self.processor.batch_decode(
                pred_output_tokens,
                skip_special_tokens=True
            )
            tik2 = time.time_ns()
            ModelMonitor.RecordTextGenerationTime(tik2-tik1, sample_ids)
            print(processed_output)

            # 发送 LoadGen 响应
            for i in range(len(qitem)):
                output = processed_output[i]
                n_tokens = len(output)
                response_array = array.array("B", output.encode("utf-8"))
                bi = response_array.buffer_info()
                response = [lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)]
                lg.QuerySamplesComplete(response)

            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                print(f"Samples run: {self.sample_counter}")

    def load_model(self):
        self.modelx = ModelLoader.load_model(self.model_name)

        self.model = self.modelx.model

        self.device = torch.device(self.device)
        if self.device == "cpu":
            self.model = self.model.to(
                self.device
            )  # Force CPU if your system has GPU and you specifically want CPU-only run

        self.model.eval()
        try:  # for systems with low ram, the below command gives error as some part is offloaded to disk
            self.model = self.model.to(memory_format=torch.channels_last)
        except BaseException:
            pass

        self.processor = self.modelx.processor
        print("Loaded processor")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def predict(self, **kwargs):
        raise NotImplementedError

    def issue_queries(self, query_samples):
        """Receives samples from loadgen and adds them to queue. Users may choose to batch here"""

        print(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[: self.batch_size])
            query_samples = query_samples[self.batch_size :]
        print(f"IssueQuery done")

    def flush_queries(self):
        pass

    def __del__(self):
        pass

class SUT_Native_SingleStream(SUT_native_base):
    def __init__(
        self,
        model_name,
        dtype,
        batch_size,
        dataset_name,
        total_sample_count,
        device,
        workers,
        test_mode,
        use_cached_outputs=False,
    ):
        SUT_native_base.__init__(
            self,
            model_name,
            dtype,
            batch_size,
            dataset_name,
            total_sample_count,
            device,
            workers,
            test_mode,
            use_cached_outputs,
        )

    """IssueQuery and inference methods implemented in Base class"""


class SUT_Native_MultiStream(SUT_native_base):
    def __init__(
        self,
        model_name,
        dtype,
        batch_size,
        dataset_name,
        total_sample_count,
        device,
        workers,
        test_mode,
        use_cached_outputs=False,
    ):
        SUT_native_base.__init__(
            self,
            model_name,
            dtype,
            batch_size,
            dataset_name,
            total_sample_count,
            device,
            workers,
            test_mode,
            use_cached_outputs,
        )

    """IssueQuery and inference methods implemented in Base class"""


class SUT_Native_Offline(SUT_native_base):
    def __init__(
        self,
        model_name,
        dtype,
        batch_size,
        dataset_name,
        total_sample_count,
        device,
        workers,
        test_mode,
        use_cached_outputs=False,
    ):
        SUT_native_base.__init__(
            self,
            model_name,
            dtype,
            batch_size,
            dataset_name,
            total_sample_count,
            device,
            workers,
            test_mode,
            use_cached_outputs,
        )

    """IssueQuery and inference methods implemented in Base class"""


class FirstTokenStreamer(BaseStreamer):
    """Streams first tokens to a 'holder'"""

    def __init__(
        self, first_token, tokens_cache=[], is_first_token=True, response_ids=[]
    ):
        """Response ids added to 'sign' the first token"""

        self.first_token = first_token  # Queue for first token
        self.is_first_token = is_first_token

        # Cache for subsequent generated tokens
        self.tokens_cache = tokens_cache

        self.response_ids = response_ids

        self.is_prompt = (
            True  # The first tokens sent to the streamer are actually the input prompts
        )

    def put(self, value):
        """Caches the tokens as they're generated. Assumes bs=1"""

        # Prompts are streamed first so we need to skip the first time value
        # that arrives
        if self.is_prompt:
            self.is_prompt = False
            return

        value = value.item()
        if self.is_first_token:

            # Add generated first token together with its query response_id to
            # first tokens queue
            self.first_token.put((value, self.response_ids[0]))

            self.is_first_token = False
            return

        self.tokens_cache.append(value)

    def end(self):
        pass

    def get_out_tokens(self):
        return self.tokens_cache


class SUT_Native_Server(SUT_native_base):
    def __init__(
        self,
        model_name,
        dtype,
        batch_size,
        dataset_name,
        total_sample_count,
        device,
        workers,
        test_mode,
        use_cached_outputs=False,
    ):
        SUT_native_base.__init__(
            self,
            model_name,
            dtype,
            batch_size,
            dataset_name,
            total_sample_count,
            device,
            workers,
            test_mode,
            use_cached_outputs,
        )
        self.test_mode = test_mode
        self.first_token_queue = queue.Queue()

    def start(self):
        if self.test_mode == lg.TestMode.AccuracyOnly:
            super().start()
            return
        
        # 创建工作线程
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

        # 创建第一个令牌响应线程
        self.ft_response_thread = threading.Thread(
            target=self.process_first_tokens)
        self.ft_response_thread.start()

    def process_first_tokens(self):
        while True:
            first_token_item = self.first_token_queue.get()

            if first_token_item is None:
                print("Exiting First token response thread")
                break

            first_tokens, response_id = first_token_item

            response_data = array.array(
                "B", np.array(
                    first_tokens, np.int32).tobytes())
            bi = response_data.buffer_info()
            response = [lg.QuerySampleResponse(response_id, bi[0], bi[1])]
            lg.FirstTokenComplete(response)

    def process_queries(self):
        if self.test_mode == lg.TestMode.AccuracyOnly:
            super().process_queries()
            return
        """处理请求队列中的查询"""
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            questions = []
            images_list = []
            sample_ids = []
            tokens_cache = []
            tokens_streamer = FirstTokenStreamer(
                self.first_token_queue,
                tokens_cache=tokens_cache,
                is_first_token=True,
                response_ids=[qitem.id],
            )
            sample = self.data_object.dataset[qitem.index]
            questions.append(sample["question"])
            images_list.extend(sample["images"])
            sample_ids.append(sample["question_id"])
                
            if len(images_list) == 0:
                images_list = None

            inputs = self.processor(
                text=questions,
                images=images_list, # 视频关键帧
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # 生成模型输出
            _ = self.model.generate(
                **inputs,
                streamer=tokens_streamer,
                **gen_kwargs
                )

            output_tokens = tokens_streamer.get_out_tokens()
            n_tokens = len(output_tokens)
            # 处理模型输出
            # processed_output = self.processor.batch_decode(
            #     output_tokens,
            #     skip_special_tokens=True
            # )
            # print(processed_output)
            response_array = array.array(
                "B", np.array(output_tokens, np.int32).tobytes()
            )
            bi = response_array.buffer_info()
            response = [
                lg.QuerySampleResponse(
                    qitem.id,
                    bi[0],
                    bi[1],
                    n_tokens)]
            lg.QuerySamplesComplete(response)
            with self.sample_counter_lock:
                self.sample_counter += 1
                print(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        if self.test_mode == lg.TestMode.AccuracyOnly:
            super().issue_queries(query_samples)
            return
        self.query_queue.put(query_samples[0])

    def stop(self):
        if self.test_mode == lg.TestMode.AccuracyOnly:
            super().stop()
            return
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

        self.first_token_queue.put(None)
        self.ft_response_thread.join()        

    """IssueQuery and inference methods implemented in Base class"""


class SUT_remote_base:
    def __init__(
        self,
        model_name=None,
        dtype="bfloat16",
        batch_size=None,
        dataset_name=None,
        total_sample_count=24576,
        device="cuda",
        api_server=None,
        api_model_name=None,
        workers=1,
        use_cached_outputs=False,
    ):

        self.model_name = model_name or "meta-llama/Llama-2-70b-chat-hf"
        self.device = device
        self.api_servers = []
        if api_server:
            self.api_servers.append(api_server)
        self.api_model_name = api_model_name
        self.device = device

        batch_size = total_sample_count
        self.batch_size = batch_size

        # dtype
        if dtype == "bfloat16":
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
        elif dtype == "float16":
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32

        if "cuda" in self.device:
            assert torch.cuda.is_available(), "torch gpu is not available, exiting..."

        self.dataset_path = dataset_name

        self.data_object = Datasets(
            model_name=self.model_name,
            dataset_path=self.dataset_name,
            total_sample_count=total_sample_count,
            device=self.device,
        )

        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=1024,
            padding_side="left",
            use_fast=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()

        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def start(self):
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

    def query_api_vllm(self, inputs, idx):
        headers = {
            "Content-Type": "application/json",
        }
        json_data = {
            "model": self.api_model_name,
            "prompt": inputs,
            "min_tokens": 1,
            "max_tokens": 1024,
        }

        response_code = 0
        print(f"Server path {self.api_servers[idx]}/v1/completions")
        while response_code != 200:
            try:
                response = requests.post(
                    f"{self.api_servers[idx]}/v1/completions",
                    headers=headers,
                    json=json_data,
                    verify=False,
                )
                response_code = response.status_code
            except Exception as e:
                print(e)
                print("connection failure")
                break
        return [resp["text"] for resp in json.loads(response.text)["choices"]]

    def api_action_handler(self, chunk, server_idx):
        output = self.query_api_vllm(chunk, server_idx)
        return output

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic"""

        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            query_ids = [q.index for q in qitem]

            fname = "q" + "_".join([str(i) for i in query_ids])
            fname = f"run_outputs/{fname}.pkl"
            _p = Path(fname)
            if self.use_cached_outputs and _p.exists():
                # Read cache
                with _p.open(mode="rb") as f:
                    d = pickle.load(f)
                processed_output = d["outputs"]
                tik1 = None
                tik2 = None
                tik3 = None
                tok = None
            else:
                # Construct / collate batch
                max_seq_len = 1024

                tik1 = time.time()

                # OpenAI-API servers don't require padding and can take input tokens
                # directly, so we build our input_ids_tensor as a jagged list
                input_ids_tensor = []
                for q in qitem:
                    # input_ids_tensor.append(self.data_object.input_ids[q.index].tolist())
                    input_ids_tensor += self.data_object.input_ids[q.index].tolist()

                # NOTE(mgoin): I don't think this has to be a torch tensor
                # input_ids_tensor = torch.cat(input_ids_tensor)

                # print(input_ids_tensor)

                assert len(input_ids_tensor) <= self.batch_size

                tik2 = time.time()

                # NOTE(mgoin): I don't think threading is necessary since we are submitting all queries in one request
                # The API server should take care of mini-batches and
                # scheduling
                if self.api_servers:
                    """
                    decoded = self.tokenizer.batch_decode(input_ids_tensor)
                    cleaned = [entry.replace('</s>','').replace('<s>','') for entry in decoded]
                    cleaned_chunks = [list(c) for c in mit.divide(len(self.api_servers), cleaned)]
                    """
                    cleaned_chunks = [input_ids_tensor]
                    with ThreadPoolExecutor(
                        max_workers=len(self.api_servers)
                    ) as executor:
                        # needs to be tested
                        output_chunks = list(
                            executor.map(
                                self.api_action_handler,
                                cleaned_chunks,
                                range(len(self.api_servers)),
                            )
                        )
                    output = []
                    for row in output_chunks:
                        output += row
                else:
                    print(
                        "Error: Specify at least one API to which the request is to be sent!"
                    )
                    exit(1)

                tik3 = time.time()

            processed_output = self.tokenizer(output)["input_ids"]
            # for i in range(len(qitem)):
            for i in range(len(processed_output)):
                # NOTE(mgoin): Not optimal to make numpy arrays just to
                # serialize
                unpadded = np.array(processed_output[i])
                n_tokens = unpadded.shape[0]
                response_array = array.array("B", unpadded.tobytes())
                bi = response_array.buffer_info()
                response = [lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)]
                lg.QuerySamplesComplete(response)

            tok = time.time()

            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                print(f"Samples run: {self.sample_counter}")
                if tik1:
                    print(f"\tBatchMaker time: {tik2 - tik1}")
                    print(f"\tInference time: {tik3 - tik2}")
                    print(f"\tPostprocess time: {tok - tik3}")
                    print(f"\t==== Total time: {tok - tik1}")
                else:
                    print(f"\tLoaded from cache: {_p}")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def predict(self, **kwargs):
        raise NotImplementedError

    def issue_queries(self, query_samples):
        """Receives samples from loadgen and adds them to queue. Users may choose to batch here"""

        list_prompts_tokens = []
        list_prompts_attn_masks = []

        print(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[: self.batch_size])
            query_samples = query_samples[self.batch_size :]
        print(f"IssueQuery done")

    def flush_queries(self):
        pass

    def __del__(self):
        pass


class SUT_Remote_SingleStream(SUT_remote_base):
    def __init__(
        self,
        model_name,
        dtype,
        batch_size,
        dataset_name,
        total_sample_count,
        device,
        api_server,
        api_model_name,
        workers,
        use_cached_outputs,
    ):
        SUT_remote_base.__init__(
            self,
            model_name,
            dtype,
            batch_size,
            dataset_name,
            total_sample_count,
            device,
            api_server,
            api_model_name,
            workers,
            use_cached_outputs,
        )

    """IssueQuery and inference methods implemented in Base class"""


class SUT_Remote_MultiStream(SUT_remote_base):
    def __init__(
        self,
        model_name,
        dtype,
        batch_size,
        dataset_name,
        total_sample_count,
        device,
        api_server,
        api_model_name,
        workers,
        use_cached_outputs,
    ):
        SUT_remote_base.__init__(
            self,
            model_name,
            dtype,
            batch_size,
            dataset_name,
            total_sample_count,
            device,
            api_server,
            api_model_name,
            workers,
            use_cached_outputs,
        )

    """IssueQuery and inference methods implemented in Base class"""


class SUT_Remote_Offline(SUT_remote_base):
    def __init__(
        self,
        model_name,
        dtype,
        batch_size,
        dataset_name,
        total_sample_count,
        device,
        api_server,
        api_model_name,
        workers,
        use_cached_outputs,
    ):
        SUT_remote_base.__init__(
            self,
            model_name,
            dtype,
            batch_size,
            dataset_name,
            total_sample_count,
            device,
            api_server,
            api_model_name,
            workers,
            use_cached_outputs,
        )

    """IssueQuery and inference methods implemented in Base class"""


class SUT_Remote_Server(SUT_remote_base):
    def __init__(
        self,
        model_name,
        dtype,
        batch_size,
        dataset_name,
        total_sample_count,
        device,
        api_server,
        api_model_name,
        workers,
        use_cached_outputs,
    ):
        SUT_remote_base.__init__(
            self,
            model_name,
            dtype,
            batch_size,
            dataset_name,
            total_sample_count,
            device,
            api_server,
            api_model_name,
            workers,
            use_cached_outputs,
        )

    """IssueQuery and inference methods implemented in Base class"""


def Get_Native_SUT(
    model_name,
    scenario,
    dtype,
    batch_size,
    dataset_name,
    total_sample_count,
    device="cuda",
    workers=1,
    test_mode=lg.TestMode.PerformanceOnly,
):
    if scenario == "Offline":
        return SUT_Native_Offline(
            model_name,
            dtype,
            batch_size,
            dataset_name,
            total_sample_count,
            device,
            workers,
            test_mode,
        )
    elif scenario == "Server":
        return SUT_Native_Server(
            model_name,
            dtype,
            batch_size,
            dataset_name,
            total_sample_count,
            device,
            workers,
            test_mode,
        )
    elif scenario == "SingleStream":
        return SUT_Native_SingleStream(
            model_name,
            dtype,
            batch_size,
            dataset_name,
            total_sample_count,
            device,
            workers,
            test_mode,
        )
    elif scenario == "MultiStream":
        return SUT_Native_MultiStream(
            model_name,
            dtype,
            batch_size,
            dataset_name,
            total_sample_count,
            device,
            workers,
            test_mode,
        )
    else:
        raise NotImplementedError


def Get_Remote_SUT(
    model_name,
    scenario,
    dtype,
    batch_size,
    dataset_name,
    total_sample_count,
    api_server,
    api_model_name,
    device="cuda",
    workers=1,
):
    if scenario == "Offline":
        return SUT_Remote_Offline(
            model_name,
            dtype,
            batch_size,
            dataset_name,
            total_sample_count,
            api_server,
            api_model_name,
            device,
            workers,
        )
    elif scenario == "Server":
        return SUT_Remote_Server(
            model_name,
            dtype,
            batch_size,
            dataset_name,
            total_sample_count,
            api_server,
            api_model_name,
            device,
            workers,
        )
    elif scenario == "SingleStream":
        return SUT_Remote_SingleStream(
            model_name,
            dtype,
            batch_size,
            dataset_name,
            total_sample_count,
            api_server,
            api_model_name,
            device,
            workers,
        )
    elif scenario == "MultiStream":
        return SUT_Remote_MultiStream(
            model_name,
            dtype,
            batch_size,
            dataset_name,
            total_sample_count,
            api_server,
            api_model_name,
            device,
            workers,
        )
    else:
        raise NotImplementedError
