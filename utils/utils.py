
def auto_configure_device_map(num_gpus, default_device):
    # visual_encoder 算4层
    # internlm_model.model.embed_tokens 占用1层
    # norm 和 lm_head 占用1层
    # transformer.layers 占用 32 层
    # 总共34层分配到num_gpus张卡上
    num_trans_layers = 32
    per_gpu_layers = 38 / num_gpus

    device_map = {
        'visual_encoder': default_device,
        'ln_vision': default_device,
        'Qformer': default_device,
        'internlm_model.model.embed_tokens': default_device,
        'internlm_model.model.norm': default_device,
        'internlm_model.lm_head': default_device,
        'query_tokens': default_device,
        'flag_image_start': default_device,
        'flag_image_end': default_device,
        'internlm_proj.weight': default_device,
        'internlm_proj.bias': default_device,
    }

    used = 6
    gpu_target = default_device
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        # print(f"gpu_target : {gpu_target}")
        # print(f"num_gpus: {num_gpus}")
        # print(f"default_device: {default_device}")
        assert gpu_target < num_gpus + default_device
        device_map[f'internlm_model.model.layers.{i}'] = gpu_target
        used += 1

    return device_map
