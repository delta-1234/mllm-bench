
from .models import *

# A dictionary mapping of model architecture to its supported model names
MODEL_LIST = {
    T5Model: ['google/flan-t5-large'],
    LlamaModel: ['llama2-7b', 'llama2-7b-chat', 'llama2-13b', 'llama2-13b-chat', 'llama2-70b', 'llama2-70b-chat', 'llama3.2-1b'],
    PhiModel: ['phi-1.5', 'phi-2'],
    PaLMModel: ['palm'],
    OpenAIModel: ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'gpt-4-0125-preview', 'gpt-3.5-turbo-0125', 'gpt-4-turbo', 'gpt-4o'],
    VicunaModel: ['vicuna-7b', 'vicuna-13b', 'vicuna-13b-v1.3'],
    UL2Model: ['google/flan-ul2'],
    GeminiModel: ['gemini-pro'],
    MistralModel: ['mistralai/Mistral-7B-v0.1', 'mistralai/Mistral-7B-Instruct-v0.1'],
    MixtralModel: ['mistralai/Mixtral-8x7B-v0.1', 'mistralai/Mixtral-8x7B-Instruct-v0.1'],
    YiModel: ['01-ai/Yi-6B', '01-ai/Yi-34B', '01-ai/Yi-6B-Chat', '01-ai/Yi-34B-Chat'],
    BaichuanModel: ['baichuan-inc/Baichuan2-7B-Base', 'baichuan-inc/Baichuan2-13B-Base',
                    'baichuan-inc/Baichuan2-7B-Chat', 'baichuan-inc/Baichuan2-13B-Chat'],
}
MODEL_LIST_VLM = {
    BLIP2Model: ['Salesforce/blip2-opt-2.7b', 'Salesforce/blip2-opt-6.7b',
                 'Salesforce/blip2-flan-t5-xl', 'Salesforce/blip2-flan-t5-xxl'],
    LLaVAModel: ['llava-1.5-7b-hf', 'llava-1.5-13b-hf'],
    GeminiVisionModel: ['gemini-pro-vision'],
    OpenAIVisionModel: ['gpt-4-vision-preview'],
    QwenVLModel: ['Qwen/Qwen-VL', 'Qwen/Qwen-VL-Chat',
                  'qwen-vl-plus', 'qwen-vl-max'],
    InternLMVisionModel: ['internlm/internlm-xcomposer2-vl-7b'],
}

SUPPORTED_MODELS = [model for model_class in MODEL_LIST.keys() for model in MODEL_LIST[model_class]]
SUPPORTED_MODELS_VLM = [model for model_class in MODEL_LIST_VLM.keys() for model in MODEL_LIST_VLM[model_class]]

class ModelLoader:

    @staticmethod
    def load_model(model_name):
        if model_name in MODEL_LIST[T5Model]:
            return T5Model(model_name)
        elif model_name in MODEL_LIST[LlamaModel]:
            return LlamaModel(model_name, max_new_tokens=256, temperature=0, device='cuda:0', dtype='auto',system_prompt=None, model_dir=None)
        elif model_name in MODEL_LIST[PhiModel]:
            return PhiModel(model_name)
        elif model_name in MODEL_LIST[PaLMModel]:
            return PaLMModel(model_name)
        elif model_name in MODEL_LIST[OpenAIModel]:
            return OpenAIModel(model_name)
        elif model_name in MODEL_LIST[VicunaModel]:
            return VicunaModel(model_name)
        elif model_name in MODEL_LIST[UL2Model]:
            return UL2Model(model_name)
        elif model_name in MODEL_LIST[GeminiModel]:
            return GeminiModel(model_name)
        elif model_name in MODEL_LIST[MistralModel]:
            return MistralModel(model_name)
        elif model_name in MODEL_LIST[MixtralModel]:
            return MixtralModel(model_name)
        elif model_name in MODEL_LIST[YiModel]:
            return YiModel(model_name)
        elif model_name in MODEL_LIST[BaichuanModel]:
            return BaichuanModel(model_name)
        elif model_name in MODEL_LIST_VLM[BLIP2Model]:
            return BLIP2Model(model_name)
        elif model_name in MODEL_LIST_VLM[LLaVAModel]:
            return LLaVAModel(model_name, max_new_tokens=128, temperature=0.7, device='cuda', dtype='float16') 
        elif model_name in MODEL_LIST_VLM[GeminiVisionModel]:
            return GeminiVisionModel(model_name)
        elif model_name in MODEL_LIST_VLM[OpenAIVisionModel]:
            return OpenAIVisionModel(model_name)
        elif model_name in MODEL_LIST_VLM[QwenVLModel]:
            return QwenVLModel(model_name)
        elif model_name in MODEL_LIST_VLM[InternLMVisionModel]:
            return InternLMVisionModel(model_name)
        else:
            # If the dataset name doesn't match any known datasets, raise an error
            raise NotImplementedError(f"Model '{model_name}' is not supported.")






