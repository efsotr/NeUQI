import torch
from torch import Tensor


@torch.compile
def find_zp_matrix_fast(x: Tensor, powern: int, weight: Tensor, norm_weight=True):
    """
    x: (..., In)
    weight: (..., In)
    Output: (..., 1)
    """
    if norm_weight:
        weight = (weight / weight.sum(dim=-1, keepdim=True)).expand_as(x)
    else:
        weight = weight.expand_as(x)
    events_R2M = torch.stack([
        - weight,
        - weight * (x - powern + 1),
        - weight * ((x - powern + 1) ** 2 - 0.25),
        x - powern + 0.5
    ])
    events_M2L = torch.stack([
        weight,
        weight * x,
        weight * (x ** 2 - 0.25),
        x + 0.5
    ])
    events_init = torch.stack([
        weight.sum(dim=-1, keepdim=True),
        (weight * (x - powern + 1)).sum(dim=-1, keepdim=True),
        (weight * (x - powern + 1) ** 2).sum(dim=-1, keepdim=True),
        torch.tensor(float("-inf"), device=x.device).expand(x.shape[:-1] + (1,))
    ])

    events = torch.cat([events_init, events_M2L, events_R2M], dim=-1)
    idx = events[3].argsort()
    events = events.gather(dim=-1, index=idx[None].expand_as(events))
    z_left, z_right = events[3][..., :-1], events[3][..., 1:]
    events = torch.cumsum(events[:3], dim=-1)
    alpha0, alpha1, alpha2 = events[0][..., :-1], events[1][..., :-1], events[2][..., :-1]
    alpha0[alpha0 == 0] = 1
    z_scope_best = torch.clamp(alpha1 / alpha0, z_left, z_right)
    obj = alpha0 * z_scope_best ** 2 - 2 * alpha1 * z_scope_best + alpha2
    z_best = z_scope_best.gather(dim=-1, index=obj.argmin(dim=-1, keepdim=True))

    z_scope_left = z_best - 1
    z_scope_right = z_best + 1

    x_add_half = x + 0.5
    x_add_half_left_dis = torch.floor(x_add_half - z_scope_left)
    x_round = x - torch.clamp(x_add_half_left_dis, 0, powern - 1)
    events_init = torch.stack([
        (weight * x_round).sum(dim=-1, keepdim=True),
        (weight * x_round ** 2).sum(dim=-1, keepdim=True),
        z_scope_left
    ])
    x_round_factor = - x_add_half_left_dis
    x_round = torch.maximum(x_add_half + x_round_factor, z_scope_left)
    round0_mask = ~((x_round <= z_scope_right) & (- powern + 1 <= x_round_factor) & (x_round_factor <= -1))
    events0 = torch.stack([
        weight,
        weight * x_round * 2,
        x_round
    ])
    events0[:2].masked_fill_(round0_mask[None], 0)

    round1_mask = ~((x_round + 1 <= z_scope_right) & (- powern + 1 <= x_round_factor + 1) & (x_round_factor + 1 <= -1))
    events1 = torch.stack([
        weight,
        weight * (x_round + 1) * 2,
        x_round + 1
    ])
    events1[:2].masked_fill_(round1_mask[None], 0)

    events_end = torch.zeros_like(events_init)
    events_end[-1] = z_scope_right
    events = torch.cat([events_init, events0, events1, events_end], dim=-1)

    idx = events[2].argsort(stable=True)
    events = events.gather(dim=-1, index=idx[None].expand_as(events))
    z_left, z_right = events[2][..., :-1], events[2][..., 1:]
    events = torch.cumsum(events[:2], dim=-1)

    alpha1, alpha2 = events[0][..., :-1], events[1][..., :-1]
    z_scope_best = torch.clamp(alpha1, z_left, z_right)
    obj = (z_scope_best - alpha1) ** 2 + (alpha2 - alpha1 * alpha1)
    obj_best, obj_idx_mn = obj.min(dim=-1, keepdim=True)
    z_best = z_scope_best.gather(dim=-1, index=obj_idx_mn)

    return z_best, obj_best

@torch.compile
def grid_zp(x: Tensor, powern: int, weight: Tensor, norm_weight=True):
    if norm_weight:
        weight = (weight / weight.sum(dim=-1, keepdim=True)).expand_as(x)
    else:
        weight = weight.expand_as(x)
    neg_zp = torch.arange(powern, device=x.device)
    _x = (x[..., None] + neg_zp)
    obj = (weight[..., None] * (_x - _x.round().clamp(0, powern-1)) ** 2).sum(dim=-2)
    obj_best, z_best = obj.min(dim=-1, keepdim=True)
    return - z_best.to(x.dtype), obj_best
