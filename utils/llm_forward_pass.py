
from transformers import AutoModel, AutoTokenizer
import torch
from utils.utils import auto_configure_device_map

# # Load the model and tokenizer
checkpoint = 'internlm/internlm-xcomposer-vl-7b'
model = AutoModel.from_pretrained(
    checkpoint, trust_remote_code=True, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True).cuda(7).eval()
model.tokenizer = tokenizer
device_map = auto_configure_device_map(
    config.num_gpus, config.default_device)
print("Configured device_map.")
print(config.default_device)
model = dispatch_model(model, device_map=device_map)
text = " this is a test run"
model(text)