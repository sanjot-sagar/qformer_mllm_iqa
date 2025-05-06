import torch
from transformers import AutoModelForCausalLM
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import *

checkpoint = "MAGAer13/mplug-owl2-llama2-7b"


model_name = get_model_name_from_path(checkpoint)
print(model_name)
# Move the model to CUDA device at index 6
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer, model, image_processor, context_len = load_pretrained_model(
    checkpoint, None, model_name,  device=device)
# model = model.to(device)

# # Print all parameter names
# print("mplug named parameters")
# for name, param in model.named_parameters():
#   print(name)
# print("=============================================================================================")
# print("mplug named children")
# for name, module in model.named_children():
#     print(name)

# print model's image_processor
print(image_processor)
