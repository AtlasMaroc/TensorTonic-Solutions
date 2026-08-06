import torch

def scaled_dot_product_attention(Q, K, V):
    """
    Returns: attention output tensor
    """

    Q = torch.tensor(Q)
    K = torch.tensor(K)
    V = torch.tensor(V)

    dot_product = Q @ torch.transpose(K, -2, -1) 

    dot_product = dot_product * 1/torch.sqrt(torch.tensor(Q.shape[-1]))

    softmax = torch.softmax(dot_product, dim=-1)
    
    return softmax @ V