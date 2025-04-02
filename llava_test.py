import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型加载
MODEL_PATH = "/home/dt/mllm-bench/model/llava-1.5-7b-hf"
processor = LlavaProcessor.from_pretrained(MODEL_PATH, use_fast=False)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float16,
).to(device)

# 加载多张图片
# image_paths = ["./image.jpg", "./image1.jpg"]
# images = [Image.open(path).convert("RGB") for path in image_paths]

# 构建多图prompt（每个<image>对应一张图）
prompt1 = """USER: 
question:Do you konw Beijing?
ASSISTANT:"""

# 处理多图输入
inputs = processor(
    text=prompt1, 
    images=None,  # 直接传入图片列表
    return_tensors="pt",
    padding=True
).to(device)

# 生成响应
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2
    )

# 解码输出
response = processor.batch_decode(outputs, skip_special_tokens=True)
print("多图响应结果：\n", response)