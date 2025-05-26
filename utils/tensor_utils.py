import torch

def test_tensor(tensor):
    """
    Utility function to print tensor statistics for debugging.
    """
    print("Shape:", tensor.shape)
    print("Average:", tensor.mean().item())
    print("Std Dev:", tensor.std().item())
    print("Min:", tensor.min().item())
    print("Max:", tensor.max().item())
    print("Contains NaN:", torch.isnan(tensor).any().item())
    print("Contains Inf:", torch.isinf(tensor).any().item())
    print("Gradient:", tensor.grad)