# from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
# import torch


# def get_all_layers(model):
#     all_layers = []
#     for name, module in model.named_children():
#         if isinstance(module, torch.nn.Module):
#             all_layers.extend(get_all_layers(module))
#         else:
#             all_layers.append((name, module))
#     return all_layers


# # Load the model and tokenizer
# checkpoint = '/home/sanjotst/.cache/huggingface/hub/models--internlm--internlm-xcomposer-vl-7b/'
# model = AutoModelForCausalLM.from_pretrained(
#     checkpoint, trust_remote_code=True, torch_dtype=torch.float32)
# tokenizer = AutoTokenizer.from_pretrained(
#     checkpoint, trust_remote_code=True)
# model.tokenizer = tokenizer

# # Get all layers of the model
# all_layers = get_all_layers(model)

# # Print all layers
# for name, module in all_layers:
#     print(1)
#     print(name, module)

from transformers import AutoModel, AutoTokenizer
import torch

def get_internLM_model():
    checkpoint = 'internlm/internlm-xcomposer-vl-7b'
    
    # Try using .from_pretrained() with explicit model class
    model = AutoModel.from_pretrained(
        checkpoint, 
        trust_remote_code=True, 
        torch_dtype=torch.float32
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True)
    
    model.tokenizer = tokenizer
    print(model)
    return model

get_internLM_model()