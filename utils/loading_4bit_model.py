
image = "/home/sanjotst/llm_iqa/llm-iqa/datasets/kadid10k/images/I23_03_02.png"
import torch, auto_gptq
from transformers import AutoModel, AutoTokenizer
from auto_gptq.modeling._base import BaseGPTQForCausalLM

auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
torch.set_grad_enabled(False)

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


# init model and tokenizer
model = InternLMXComposer2QForCausalLM.from_quantized(
  'internlm/internlm-xcomposer2-vl-7b-4bit', trust_remote_code=True, device="cuda:2").eval()
tokenizer = AutoTokenizer.from_pretrained(
  'internlm/internlm-xcomposer2-vl-7b-4bit', trust_remote_code=True)

text = '<ImageHere>Please describe this image in detail.'
with torch.cuda.amp.autocast():
  response, _ = model.chat(tokenizer, query=text, image=image, history=[], do_sample=False)
print("===========================================================================================================")
print("response of quantised model is")
print(response)
print("===========================================================================================================")

##############################################
# code for internlmv2
# init model and tokenizer
model2 = AutoModel.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True)

text = '<ImageHere>Please describe this image in detail.'
with torch.cuda.amp.autocast():
  response, _ = model2.chat(tokenizer, query=text, image=image, history=[], do_sample=False)
  
print("===========================================================================================================")
print("response of internlm v2 model is")
print(response)
print("===========================================================================================================")
