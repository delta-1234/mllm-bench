import torch
from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import get_model_name_from_path, process_images
from llava.model.builder import load_pretrained_model
from PIL import Image

# 初始化配置（强制使用单一设备）
torch.set_grad_enabled(False)
model_path = "/data/dengtao/model/llava-v1.5-7b"
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

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
image_file = "image1.jpg"
question = "Describe this image in detail."
image = Image.open(image_file).convert("RGB")

# 图像预处理（设备统一）
try:
    image_size = image_processor.crop_size['height']
    processed_image = image_processor.preprocess(
        image, 
        return_tensors='pt',
        size=image_size,
        crop_size=image_size
    )['pixel_values'][0]
    vision_inputs = processed_image.unsqueeze(0).to(device, dtype=torch.float16)
except Exception as e:
    raise RuntimeError(f"图像处理失败: {str(e)}") from e

# 构建prompt（设备统一）
prompt = f"USER: <image>\n{question}\nASSISTANT:"
input_ids = tokenizer(
    prompt, 
    return_tensors="pt", 
    max_length=tokenizer.model_max_length,
    truncation=True
).input_ids.to(device)  # 显式指定设备

# 创建CUDA事件
def create_events():
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True)
    )

# 初始化计时事件
vision_start, vision_end = create_events()
cross_start, cross_end = create_events()
text_start, text_end = create_events()

# 视觉编码
vision_start.record()
with torch.inference_mode():
    vision_features = model.model.vision_tower(vision_inputs)
vision_end.record()

# 跨模态对齐
cross_start.record()
with torch.inference_mode():
    # 强制投影层输出到目标设备
    projected_vision = model.model.mm_projector(vision_features).to(device)
    # 显式移动文本嵌入到目标设备
    text_embeddings = model.model.embed_tokens(input_ids).to(device)
    
    # 维度对齐
    num_vision_patches = vision_features.shape[1]
    combined_embeddings = torch.cat([
        projected_vision[:, :num_vision_patches],
        text_embeddings[:, 1:]  # 跳过BOS token
    ], dim=1).to(device)  # 最终确认设备
cross_end.record()

# 文本生成
text_start.record()
with torch.inference_mode():
    outputs = model(
        inputs_embeds=combined_embeddings,
        output_attentions=True,
        use_cache=False
    )
text_end.record()

# 同步并计算时间
torch.cuda.synchronize()

print(f"\n=== 推理时间分析 ===")
print(f"Vision Encoder Time: {vision_start.elapsed_time(vision_end):.2f} ms")
print(f"Cross-modal Alignment Time: {cross_start.elapsed_time(cross_end):.2f} ms")
print(f"Text Generation Time: {text_start.elapsed_time(text_end):.2f} ms")
