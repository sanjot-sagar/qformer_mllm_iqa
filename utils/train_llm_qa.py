import torch
import time


def occupy_gpu_memory(device_id, memory_to_use):
    torch.cuda.set_device(device_id)
    # Each element is 4 bytes (float32)
    tensor_size = (memory_to_use * 1024 ** 3) // 4
    tensor = torch.zeros((tensor_size,), dtype=torch.float32).cuda()
    # print(f"GPU {device_id}: Occupied {memory_to_use} GB of memory")


def main():
    # Specify the GPU indices you want to use
    gpu_indices = [5]
    memory_per_gpu = 41  # GB
    assert memory_per_gpu <= 49, "Requested memory per GPU exceeds available memory"

    try:
        while True:
            for gpu_index in gpu_indices:
                occupy_gpu_memory(gpu_index, memory_per_gpu)
            time.sleep(1)  # Add a small delay to reduce CPU load
    except KeyboardInterrupt:
        print("Execution interrupted. Stopping memory occupation.")


if __name__ == "__main__":
    main()
