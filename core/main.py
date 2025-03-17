import subprocess
import mlperf_loadgen as lg
import argparse
import logging
import os
import math
import sys

from pathlib import Path
from pprint import pprint

try:
    project_root = Path(__file__).parent.parent.resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"Project root directory not found: {project_root}")
    sys.path.append(str(project_root))
except Exception as e:
    print(f"Error setting up project path: {e}")
    sys.exit(1)


from dataload import SUPPORTED_DATASETS, SUPPORTED_DATASETS_VLM
from modeload import SUPPORTED_MODELS, SUPPORTED_MODELS_VLM

from Sever_SUT import Get_Sever_SUT
from Native_SUT import Get_Native_SUT,Get_Remote_SUT
from QDL import QDL
from QSL import get_QSL

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

SUPPORTED_BACKEND = {"hf": "pytorch", "vllm": "vllm", "trtllm": "trtllm"}

SUPPORTED_PROFILES = {
    "hf": {
        "dataset": "SEED-Bench-2",
        "backend": "pytorch",
        "model-name": "llava-1.5-7b-hf",
    },
    "vllm": {
        "dataset": "mmlu",
        "backend": "vllm",
        "model-name": "llama3.2-3b",
    },
}

def get_temp_cache():
    beam_size = int(os.environ.get("GPTJ_BEAM_SIZE", "4"))
    return 6 * beam_size


def get_args():
    parser = argparse.ArgumentParser(description='This is a program that demonstrates the usage of Argparse.')

    parser.add_argument(
        "--scenario",
        type=str,
        choices=SCENARIO_MAP.keys(),
        default="Offline",
        help="benchmark scenario, one of" + str(list(SCENARIO_MAP.keys())),
    )

    parser.add_argument(
        "--backend",
        choices=SUPPORTED_BACKEND.keys(),
        default="hf",
        help="backend"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Model path",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model Name",
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="dataset path"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset"
    )

    parser.add_argument(
        "--dtype",
        default="float32",
        help="data type of the model, choose from float16, bfloat16 and float32",
    )

    parser.add_argument(
        "--profile",
        choices=SUPPORTED_PROFILES.keys(),
        help="standard profiles",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="device to use",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="use GPU instead of CPU for the inference"
    )

    parser.add_argument(
        "--audit_conf",
        default="audit.conf",
        help="audit config for LoadGen settings during compliance runs",
    )

    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )

    parser.add_argument(
        "--total-sample-count",
        type=int,
        default=24576,
        help="Number of samples to use in benchmark.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Model batch-size to use in benchmark.",
    )

    parser.add_argument(
        "--max-batchsize",
        type=int,
        default=1,
        help="max batch size in a single inference",
    )

    parser.add_argument(
        "--output-log-dir",
        type=str,
        default="output-logs",
        help="Where logs are saved"
    )

    parser.add_argument(
        "--max_examples",
        type=int,
        default=13368,
        help="Maximum number of examples to consider (not limited by default)"
    )

    parser.add_argument(
        "--enable-log-trace",
        action="store_true",
        help="Enable log tracing. This file can become quite large",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers to process queries",
    )

    parser.add_argument(
        "--network",
        choices=["sut", "lon", "edge"],
        default="edge",
        help="Loadgen network mode",
    )

    parser.add_argument(
        "--node",
        type=str,
        default="",
        help="mutilGPU node mode",
    )

    # sut mode
    parser.add_argument(
        "--port",
        type=int,
        default=8000
    )

    # lon mode
    parser.add_argument(
        "--sut_server",
        nargs="*",
        default=["http://localhost:8000"],
        help="Address of the server(s) under test.",
    )

    #edge mode
    parser.add_argument(
        "--mode",
        choices=["native", "remote"],
        default="native",
        help="generate mode",
    )

    #remote mode
    parser.add_argument(
        "--api-model-name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model name(specified on remote mode)",
    )

    parser.add_argument(
        "--api-server",
        type=str,
        default=None,
        help="Specify an api endpoint call to use api mode",
    )

    return parser.parse_args()


def show_support(args):

    if args.dataset == "?":
        print("LLM Dataset:")
        max_width = max(len(name) for name in SUPPORTED_DATASETS)
        for i in range(0, len(SUPPORTED_DATASETS), 5):
            row = [f"{name:<{max_width}}" for name in SUPPORTED_DATASETS[i:i+5]]
            print("  ".join(row))
        print("MLLM Dataset:")
        max_width = max(len(name) for name in SUPPORTED_DATASETS_VLM)
        for i in range(0, len(SUPPORTED_DATASETS_VLM), 5):
            row = [f"{name:<{max_width}}" for name in SUPPORTED_DATASETS_VLM[i:i+5]]
            print("  ".join(row))
        sys.exit(0)

    if args.model_name == "?":
        print("LLM:")
        max_width = max(len(name) for name in SUPPORTED_MODELS)
        for i in range(0, len(SUPPORTED_MODELS), 5):
            row = [f"{name:<{max_width}}" for name in SUPPORTED_MODELS[i:i+5]]
            print("  ".join(row))
        print("MLLM:")
        max_width = max(len(name) for name in SUPPORTED_MODELS_VLM)
        for i in range(0, len(SUPPORTED_MODELS_VLM), 5):
            row = [f"{name:<{max_width}}" for name in SUPPORTED_MODELS_VLM[i:i+5]]
            print("  ".join(row))
        sys.exit(0)


def main():
    try:
        args = get_args()
        show_support(args)
        qsl = None
        if args.network == "sut":
            print("Run sut mode...")

            sut = Get_Sever_SUT(
                model_path=args.model_path,
                scenario=args.scenario,
                dtype=args.dtype,
                use_gpu=(args.device == "cuda" or args.gpu),
                network=args.network,
                dataset_path=args.dataset_path,
                max_examples=args.max_examples,
                qsl=qsl,
            )

            temp_cache = get_temp_cache()
            from network_SUT import app, node, set_backend, set_semaphore
            from systemStatus import get_cpu_memory_info, get_gpu_memory_info

            model_mem_size = sut.total_mem_size
            if args.gpu:
                free_mem = int(
                    os.environ.get(
                        "CM_CUDA_DEVICE_PROP_GLOBAL_MEMORY", get_gpu_memory_info()
                    )
                ) / (1024**3)
            else:
                free_mem = get_cpu_memory_info()

            lockVar = math.floor((free_mem - model_mem_size) / temp_cache)
            node = args.node

            set_semaphore(lockVar)
            print(f"Set the semaphore lock variable to {lockVar}")

            set_backend(sut)
            app.run(debug=False, port=args.port, host="0.0.0.0")

            print("Destroying SUT...")
            lg.DestroySUT(sut.sut)

        elif args.network == "lon":
            print("Run lon mode...")

            qsl = get_QSL(dataset_path=args.dataset_path, max_examples=args.max_examples)
            qdl = QDL(sut_server_addr=args.sut_server, scenario=args.scenario, qsl=qsl)

            settings = lg.TestSettings()
            settings.scenario = SCENARIO_MAP[args.scenario]
            settings.FromConfig(args.user_conf, args.model_name.title(), args.scenario)
            settings.mode = lg.TestMode.PerformanceOnly

            # set logs
            os.makedirs(args.output_log_dir, exist_ok=True)
            log_output_settings = lg.LogOutputSettings()
            log_output_settings.outdir = args.output_log_dir
            log_output_settings.copy_summary_to_stdout = True
            log_settings = lg.LogSettings()
            log_settings.log_output = log_output_settings
            log_settings.enable_trace = args.enable_log_trace

            lg.StartTestWithLogSettings(
                qdl.qdl, qsl.qsl, settings, log_settings, args.audit_conf
            )

            print("Destroying QSL...")
            lg.DestroyQSL(qsl.qsl)

        elif args.network == "edge":

            print("Run edge mode...")
            settings = lg.TestSettings()
            settings.scenario = SCENARIO_MAP[args.scenario]
            settings.FromConfig(args.user_conf, args.model_name, args.scenario)
            settings.mode = lg.TestMode.PerformanceOnly
            print(11111)
            print(args.model_name)
            # set logs
            os.makedirs(args.output_log_dir, exist_ok=True)
            log_output_settings = lg.LogOutputSettings()
            log_output_settings.outdir = args.output_log_dir
            log_output_settings.copy_summary_to_stdout = True
            log_settings = lg.LogSettings()
            log_settings.log_output = log_output_settings
            log_settings.enable_trace = args.enable_log_trace

            if args.mode == "native":
                sut = Get_Native_SUT(
                    model_name=args.model_name,
                    scenario=args.scenario,
                    dtype=args.dtype,
                    batch_size=args.batch_size,
                    dataset_name=args.dataset,
                    total_sample_count=args.total_sample_count,
                    device=args.device,
                    workers=args.num_workers,
                )
            elif args.mode == "remote":
                sut = Get_Remote_SUT(
                    model_name=args.model_name,
                    scenario=args.scenario,
                    dtype=args.dtype,
                    batch_size=args.batch_size,
                    dataset_name=args.dataset,
                    total_sample_count=args.total_sample_count,
                    api_server=args.api_server,
                    api_model_name=args.api_model_name,
                    device=args.device,
                    workers=args.num_workers,
                )
            else:
                raise NotImplementedError

            # Start sut before loadgen starts
            sut.start()
            print("native start...")
            lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)

            lg.StartTestWithLogSettings(
                lgSUT, sut.qsl, settings, log_settings, args.audit_conf
            )

            sut.stop()
            print("native stop...")

            lg.DestroySUT(lgSUT)
            lg.DestroyQSL(sut.qsl)

        else:
            print("Not support mode...")

        print("Test Done!")

    except Exception as e:
        print(f"An error occurred: {e}")
        print('Program is dead.')
    finally:
        print('Clean Program')

if __name__ == "__main__":
    main()
