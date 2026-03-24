import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    d=K.shape[2]
    att_scores=torch.matmul(Q,torch.transpose(K,1,2))/math.sqrt(d)
    att_weights=F.softmax(att_scores,dim=-1)
    res=torch.matmul(att_weights,V)
    return res
    pass