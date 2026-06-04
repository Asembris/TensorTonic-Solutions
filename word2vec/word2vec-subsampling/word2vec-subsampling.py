import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    counts=counts.float()
    freqs=counts/counts.sum()
    probs=torch.sqrt(t/freqs)
    probs=torch.clamp(probs,max=1.0)
    return probs
    pass
