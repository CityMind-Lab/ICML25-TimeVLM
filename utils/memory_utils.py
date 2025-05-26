import torch
import gc

def test_cuda_memory(device):
    """
    Test CUDA memory usage.

    Args:
    - device: CUDA device.
    """
    print(torch.cuda.memory_summary(device=device, abbreviated=False))
    max_memory = torch.cuda.max_memory_allocated()
    print(f"Max memory allocated: {max_memory / (1024 ** 3):.2f} GB")
    # Force garbage collection
    gc.collect()