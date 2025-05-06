import random
import numpy as np
import torch


def auto_configure_device_map_v2(num_gpus, default_device):
    # visual_encoder 算4层
    # internlm_model.model.embed_tokens 占用1层
    # norm 和 lm_head 占用1层
    # transformer.layers 占用 32 层
    # 总共34层分配到num_gpus张卡上
    num_trans_layers = 32
    print("num gpus")
    print(num_gpus)
    per_gpu_layers = 38 / num_gpus

    last_gpu = default_device + num_gpus - 1
    device_map = {
        'vit': default_device,
        'vision_proj': default_device,
        'model.tok_embeddings': default_device,
        'model.norm': last_gpu,
        'output': last_gpu,
    }

    used = 3
    gpu_target = default_device
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus + default_device
        device_map[f'model.layers.{i}'] = gpu_target
        used += 1

    return device_map


# def auto_configure_device_map_v2(num_gpus):
#     # visual_encoder 算4层
#     # internlm_model.model.embed_tokens 占用1层
#     # norm 和 lm_head 占用1层
#     # transformer.layers 占用 32 层
#     # 总共34层分配到num_gpus张卡上
#     num_trans_layers = 32
#     per_gpu_layers = 38 / num_gpus

#     device_map = {
#         'vit': 0,
#         'vision_proj': 0,
#         'model.tok_embeddings': 0,
#         'model.norm': num_gpus - 1,
#         'output': num_gpus - 1,
#     }

#     used = 3
#     gpu_target = 0
#     for i in range(num_trans_layers):
#         if used >= per_gpu_layers:
#             gpu_target += 1
#             used = 0
#         assert gpu_target < num_gpus
#         device_map[f'model.layers.{i}'] = gpu_target
#         used += 1

#     return device_map
