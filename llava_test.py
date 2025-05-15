import torch
from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from PIL import Image

# 初始化配置（强制使用单一设备）
torch.set_grad_enabled(False)
model_path = "/home/buaa/dengtao/model/llava-v1.5-7b"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 验证模型
model_name = get_model_name_from_path(model_path)
assert "llava" in model_name.lower(), "必须使用LLaVA多模态检查点！"

print(f"运行设备: {device}")
print(f"加载模型: {model_name}")

# 加载模型（强制指定设备）
try:
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device_map=None,  # 禁用自动设备映射
    )
    model = model.to(device)  # 二次确认设备位置
    model.eval()
except Exception as e:
    raise RuntimeError(f"模型加载失败: {str(e)}") from e

# 准备输入（统一设备）
image_file = "image.jpg"
question = "Describe this image in detail."
image = Image.open(image_file).convert("RGB")

start_vision = torch.cuda.Event(enable_timing=True) # GPU精准计时
end_vision = torch.cuda.Event(enable_timing=True)
start_vision.record()
try:
    image_tensor = process_images([image], image_processor, model.config)[0]
except Exception as e:
    raise RuntimeError(f"图像处理失败: {str(e)}") from e
end_vision.record()
torch.cuda.synchronize()
vision_time = start_vision.elapsed_time(end_vision)

# 构建prompt
start_text = torch.cuda.Event(enable_timing=True) # GPU精准计时
end_text = torch.cuda.Event(enable_timing=True)
start_text.record()
prompt = f"USER: <image>\n{question}\nASSISTANT:"
input_ids = tokenizer_image_token(prompt, tokenizer, -200, return_tensors='pt').unsqueeze(0).cuda()
end_text.record()
torch.cuda.synchronize()
text_time = start_text.elapsed_time(end_text)

start_generation = torch.cuda.Event(enable_timing=True)
end_generation = torch.cuda.Event(enable_timing=True)
start_generation.record()
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor.unsqueeze(0).half().cuda(),
        image_sizes=[image.size],
        do_sample=True,
        # no_repeat_ngram_size=3,
        max_new_tokens=512,
        use_cache=True)
end_generation.record()
torch.cuda.synchronize()
generation_time = start_generation.elapsed_time(end_generation)

outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(outputs)
cross_modal_wait = end_vision.elapsed_time(start_generation)

print(f"\n[性能指标]")
print(f"视觉处理时间: {vision_time:.3f}ms")
print(f"文本处理时间: {text_time:.3f}ms")
print(f"文本生成时间: {generation_time:.3f}ms")
print(f"跨模态等待时间: {cross_modal_wait:.3f}ms")
