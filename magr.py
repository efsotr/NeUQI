import torch
from torch import Tensor

@torch.compile(dynamic=True)
def project_onto_l1_ball_groupwise(x: Tensor, eps=1.0):
    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    
    v = torch.sort(torch.abs(x), dim=1, descending=True).values
    cumsum = torch.cumsum(v, dim=1)
    idx = torch.arange(1, x.size(-1) + 1, device=x.device, dtype=torch.int)
    rho = torch.max((v * idx > (cumsum - eps)) * idx, dim=1, keepdim=True).indices
    theta = ((cumsum.gather(dim=1, index=rho) - eps) / (rho + 1)).clamp(min=0)
    x = (torch.abs(x) - theta).clamp(min=0) * torch.sign(x)
    
    return x.view(x_shape)

@torch.compile(dynamic=True)
def linfty_proximal_groupwise(x, alpha, group_size):
    x_group = x.view(-1, group_size)
    proximal_result = x_group - alpha * project_onto_l1_ball_groupwise(x_group / alpha)
    return proximal_result.view(x.shape)

def W_proximal_preprocess_groupwise(
    H: Tensor,
    W: Tensor, 
    group_size: int, 
    alpha = None,
    default_alpha=1e-5, 
    n_iter=150
):
    if group_size == -1:
        group_size = H.size(0)
        default_alpha = 1e-3
    if alpha is None:
        alpha = default_alpha
    val = torch.linalg.eigvalsh(H)
    H = H / val.max()

    W_hat = W
    for _ in range(n_iter):
        W_hat = linfty_proximal_groupwise(
            W_hat - torch.matmul(W_hat - W, H), 
            alpha=alpha, 
            group_size=group_size
        )

    return W_hat