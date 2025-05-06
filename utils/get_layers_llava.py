# from transformers import AutoProcessor, LlavaForConditionalGeneration
# import torch

# # Set GPU device (replace 6 with your desired GPU index)
# device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# # check point
# model_name = "llava-hf/llava-1.5-7b-hf"
# print(model_name)
# # Download the model
# model = LlavaForConditionalGeneration.from_pretrained(model_name)

# # Move the model to the specified GPU
# model.to(device)


# # Print all parameter names
# print("llava named parameters")
# for name, param in model.named_parameters():
#   print(name)
# print("=============================================================================================")
# print("llava named children")
# for name, module in model.named_children():
#     print(name)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"
model_name = get_model_name_from_path(model_path)
print(model_name)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name
)
print(model_name)

# # Print all parameter names
# print("llava named parameters")
# for name, param in model.named_parameters():
#   print(name)
# print("=============================================================================================")
# print("llava named children")
# for name, module in model.named_children():
#     print(name)

# print model's image_processor
print(image_processor)
