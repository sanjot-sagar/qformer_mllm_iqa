from transformers import AutoModel, AutoTokenizer
import torch
# imports for quantisation
import torch
import auto_gptq
from transformers import AutoModel, AutoTokenizer
from auto_gptq.modeling._base import BaseGPTQForCausalLM
from accelerate import dispatch_model

# InternLMv2 model device map import
from device_map_internlm_v2 import auto_configure_device_map_v2


# # Load the model and tokenizer
checkpoint = 'internlm/internlm-xcomposer-vl-7b'
model = AutoModel.from_pretrained(
    checkpoint, trust_remote_code=True, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model.tokenizer = tokenizer


# Print the names of all modules in the model
print("========================================================================================================================================================")
print("internlm-xcomposer-vl-7b")
for name, module in model.named_children():
    print(name)
print("========================================================================================================================================================")
print("printing parameters of the model")
for name, param in model.named_parameters():
    print(name)

#################################################################################################
# quantised internlm model class


class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
    layers_block_name = "model.layers"
    outside_layer_modules = [
        'vit', 'vision_proj', 'model.tok_embeddings', 'model.norm', 'output',
    ]
    inside_layer_modules = [
        ["attention.wqkv.linear"],
        ["attention.wo.linear"],
        ["feed_forward.w1.linear", "feed_forward.w3.linear"],
        ["feed_forward.w2.linear"],
    ]


checkpoint = 'internlm/internlm-xcomposer2-vl-7b-4bit'
model = InternLMXComposer2QForCausalLM.from_quantized(
    checkpoint, trust_remote_code=True, torch_dtype=torch.float32, device="cuda:0").cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint, trust_remote_code=True)
model.tokenizer = tokenizer

# Print the names of all modules in the model
print("=============================================================================================================================================")
print("printing modules of quantised model")
for name, module in model.named_children():
    print(name)
print("========================================================================================================================================================")
print("printing parameters of the model")
for name, param in model.named_parameters():
    print(name)

###################################################################################################################

checkpoint = 'internlm/internlm-xcomposer2-vl-7b'
model = AutoModel.from_pretrained(
    checkpoint, trust_remote_code=True, torch_dtype=torch.float32).cuda().eval()
num_gpus = 2
device_map = auto_configure_device_map_v2(num_gpus, default_device=0)
print("Configured device_map.")
model = dispatch_model(model, device_map=device_map)
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint, trust_remote_code=True)
model.tokenizer = tokenizer

# Print the names of all modules in the model
print("========================================================================================================================================================")
print("printing modules of the v2 model")
for name, module in model.named_children():
    print(name)
print("========================================================================================================================================================")
print("printing parameters of the model")
for name, param in model.named_parameters():
    print(name)
