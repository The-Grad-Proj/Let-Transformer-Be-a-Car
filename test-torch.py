import torch
import os

print(torch.cuda.get_device_name(0))  # Name of your GPU
print(torch.cuda.memory_allocated(0))  # Memory used by tensors
print(torch.cuda.memory_reserved(0))  # Memory reserved by the caching allocator
print(torch.cuda.get_device_properties(0).total_memory)  # Total GPU memory

num_workers = os.cpu_count()
print(num_workers)