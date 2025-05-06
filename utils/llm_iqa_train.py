import torch
import time

# Calculate total usable memory across all GPUs (assuming 48 GB each)
total_usable_memory = 48 * 1024**3  # 48 GB in bytes per GPU

# Target memory usage percentage (adjusted for safety, can be fine-tuned)
target_usage_percentage = 0.85

# Calculate target memory allocation per GPU
target_memory_per_gpu = int(total_usable_memory * target_usage_percentage / 8)

# Create a large tensor on each GPU
devices = [torch.device(f"cuda:{i}") for i in range(8)]
tensors = [torch.rand(target_memory_per_gpu, dtype=torch.float32).to(
    device) for device in devices]

# Print memory usage information
for i, device in enumerate(devices):
    allocated_memory = torch.cuda.memory_allocated(device)
    cached_memory = torch.cuda.memory_cached(device)
    print(f"GPU {i}: Allocated memory: {allocated_memory / 1024**3:.2f} GB, Cached memory: {cached_memory / 1024**3:.2f} GB")

# Tolerance for memory usage fluctuations
tolerance = 0.05  # Allow 5% headroom above target

while True:
    for i, device in enumerate(devices):
        allocated_memory = torch.cuda.memory_allocated(device)
        if allocated_memory / 1024**3 > target_memory_per_gpu * (1 + tolerance):
            # Memory usage exceeded target with tolerance
            print(f"GPU {i}: Memory usage exceeded target. Adjusting...")

            # **Options for handling memory overage (choose one):**

            # Option 1: Reduce tensor size slightly (adjust factor as needed)
            # reduction_factor = 0.01
            # tensors[i] = tensors[i][:int(target_memory_per_gpu * (1 - reduction_factor))]

            # Option 2: Offload some tensors to CPU (if feasible)
            # if torch.cuda.is_available():
            #     cpu_tensor = tensors[i].cpu()
            #     del tensors[i]
            #     # (Logic to reload tensor from CPU as needed)
            # else:
            #     print(f"GPU {i}: Offloading to CPU not possible. Consider reducing tensor size.")

    # Briefly sleep to avoid busy-waiting
    time.sleep(0.1)
