from abc import ABC, abstractmethod
from functools import cache
from typing import Literal, cast

import torch
import torch._C
import torch.compiler
from torch import Tensor
from tqdm import tqdm

from arguments import PARAM_INIT_METHODS, QuantArgs
from find_zeropoint import find_zp_matrix_fast, grid_zp
from optimized_module.ops.quantize.utils_quantize import pack


torch.set_num_threads(32)

LAYER_IDX = -1
LINEAR_LAYER_NAME = ""

@cache
def info_once(message):
    print(message, flush=True)

def set_layer_idx(idx):
    global LAYER_IDX
    LAYER_IDX = idx

def set_layer_name(name):
    global LINEAR_LAYER_NAME
    LINEAR_LAYER_NAME = name

@torch.compile()
def quant(w: Tensor, scale: Tensor, zero_point: Tensor, clamp_min: int, clamp_max: int):
    return (torch.clamp(torch.round(w / scale - zero_point), clamp_min, clamp_max) + zero_point) * scale

@torch.compile()
def quant_random(w: Tensor, eta: Tensor, scale: Tensor, zero_point: Tensor, clamp_min: int, clamp_max: int):
    return (torch.clamp(torch.floor(w / scale - zero_point + eta), clamp_min, clamp_max) + zero_point) * scale

@torch.compile()
def quant_int(w: Tensor, scale: Tensor, zero_point: Tensor, clamp_min: int, clamp_max: int):
    return torch.clamp(torch.round(w / scale - zero_point), clamp_min, clamp_max)

@torch.compile()
def quant_int_z(w: Tensor, zero_point: Tensor, clamp_min: int, clamp_max: int):
    return torch.clamp(torch.round(w - zero_point), clamp_min, clamp_max)

@torch.compile()
def dquant_int(w: Tensor, scale: Tensor, zero_point: Tensor):
    return (w + zero_point) * scale

class UniformQuantizer:

    def __init__(self, quant_args: QuantArgs):
        self.nbits = quant_args.nbits
        self.max : int = 2 ** self.nbits - 1
        self.min = 0
        self.quant_args = quant_args

    @torch.no_grad()
    def init_param(self, w: torch.Tensor, H_diag: torch.Tensor | None = None):
        """
        w shape: (Out, In)
        """
        assert len(w.shape) == 2
        group_size = self.quant_args.group_size if self.quant_args.group_size > 0 else w.size(1)
        assert w.size(1) % group_size == 0
        num_groups = w.size(1) // group_size
        w = w.view(-1, num_groups, group_size)
        out_dim = w.size(0)
        global LAYER_IDX, LINEAR_LAYER_NAME

        wmax = w.max(dim=-1, keepdim=True).values # (Out, 1)
        wmin = w.min(dim=-1, keepdim=True).values

        scale = (wmax - wmin) / self.max * self.quant_args.scale_clip_beta # (Out, Num Group, 1)
        scale[scale < 1e-9] = 1
        if H_diag is not None:
            H_diag = H_diag.view(1, num_groups, group_size)
            H_diag /= H_diag.sum(dim=-1, keepdim=True)

        if self.quant_args.param_init_method_diag == PARAM_INIT_METHODS.grid_s_zp.value:
            H_diag = H_diag.view(1, num_groups, group_size)
            T = self.quant_args.grid_tot_steps
            obj_best = torch.tensor(float("inf"), device=w.device).expand(out_dim, num_groups, 1) # (Out, Num group, 1)
            scale_best = torch.zeros((out_dim, num_groups, 1), device=w.device)
            zero_point_best = torch.zeros((out_dim, num_groups, 1), device=w.device)

            range_list = torch.arange(self.quant_args.grid_start_steps, T + 1) / T
            if self.quant_args.enable_grid_low_precision_step:
                range_list = range_list.bfloat16().float().unique()
            range_list = range_list.tolist()
            loop_wraper = (lambda x: tqdm(x, desc=f"iter (init {LINEAR_LAYER_NAME})", miniters=128)) if LAYER_IDX == 0 else (lambda x: x)

            for i in loop_wraper(range_list):
                scale_ex = scale * i # (Out, Num group, 1)
                zero_point, obj = grid_zp(w / scale_ex, 2 ** self.nbits, H_diag, norm_weight=False) # (Out, Num Group, 1)
                obj *= scale_ex.square()
                update_mask = obj < obj_best
                obj_best = torch.minimum(obj_best, obj)
                zero_point_best[update_mask] = zero_point[update_mask]
                scale_best[update_mask] = scale_ex[update_mask]

            scale, zero_point = scale_best, zero_point_best
            aux_loss = obj_best.sum().item()

        elif self.quant_args.param_init_method_diag == PARAM_INIT_METHODS.grid_s_best_zp.value:
            H_diag = H_diag.view(1, num_groups, group_size)
            T = self.quant_args.grid_tot_steps
            obj_best = torch.tensor(float("inf"), device=w.device).expand(out_dim, num_groups, 1) # (Out, Num group, 1)
            scale_best = torch.zeros((out_dim, num_groups, 1), device=w.device)
            zero_point_best = torch.zeros((out_dim, num_groups, 1), device=w.device)

            range_list = torch.arange(self.quant_args.grid_start_steps, T + 1) / T
            if self.quant_args.enable_grid_low_precision_step:
                range_list = range_list.bfloat16().float().unique()
            range_list = range_list.tolist()
            loop_wraper = (lambda x: tqdm(x, desc=f"iter (init {LINEAR_LAYER_NAME})", miniters=128)) if LAYER_IDX == 0 else (lambda x: x)

            for i in loop_wraper(range_list):
                scale_ex = scale * i # (Out, Num group, 1)
                zero_point, obj = find_zp_matrix_fast(w / scale_ex, 2 ** self.nbits, H_diag, norm_weight=False) # (Out, Num Group, 1)
                obj *= scale_ex.square()
                update_mask = obj < obj_best
                obj_best = torch.minimum(obj_best, obj)
                zero_point_best[update_mask] = zero_point[update_mask]
                scale_best[update_mask] = scale_ex[update_mask]

            scale, zero_point = scale_best, zero_point_best
            aux_loss = obj_best.sum().item()

        elif self.quant_args.param_init_method_diag == PARAM_INIT_METHODS.grid2_s_best_zp.value:
            H_diag = H_diag.view(1, num_groups, group_size)
            T = self.quant_args.grid_tot_steps
            Tmid = self.quant_args.grid_mid_tot_steps
            obj_best = torch.tensor(float("inf"), device=w.device).expand(out_dim, num_groups, 1) # (Out, Num group, 1)
            scale_best = torch.zeros((out_dim, num_groups, 1), device=w.device)
            zero_point_best = torch.zeros((out_dim, num_groups, 1), device=w.device)

            range_list = torch.arange(max(int(self.quant_args.grid_start_steps / T * Tmid), 1), Tmid + 1) / Tmid
            # if self.quant_args.enable_grid_low_precision_step:
            #     range_list = range_list.bfloat16().float().unique()
            range_list = range_list.tolist()
            loop_wraper = (lambda x: tqdm(x, desc=f"iter (init {LINEAR_LAYER_NAME})", miniters=128)) if LAYER_IDX == 0 else (lambda x: x)

            for i in loop_wraper(range_list):
                scale_ex = scale * i # (Out, Num group, 1)
                zero_point, obj = find_zp_matrix_fast(w / scale_ex, 2 ** self.nbits, H_diag, norm_weight=False) # (Out, Num Group, 1)
                obj *= scale_ex.square()
                update_mask = obj < obj_best
                obj_best = torch.minimum(obj_best, obj)
                zero_point_best[update_mask] = zero_point[update_mask]
                scale_best[update_mask] = scale_ex[update_mask]

            scale_mid = scale_best.clone()

            range_list = torch.arange(- (T // Tmid) // 2, (T // Tmid) // 2 + 1) / T
            range_list = range_list.tolist()
            loop_wraper = (lambda x: tqdm(x, desc=f"iter (init {LINEAR_LAYER_NAME})", miniters=128)) if LAYER_IDX == 0 else (lambda x: x)
            for i in loop_wraper(range_list):
                if i == 0:
                    continue
                scale_ex = scale_mid + scale * i # (Out, Num group, 1)
                zero_point, obj = find_zp_matrix_fast(w / scale_ex, 2 ** self.nbits, H_diag, norm_weight=False) # (Out, Num Group, 1)
                obj *= scale_ex.square()
                update_mask = obj < obj_best
                obj_best = torch.minimum(obj_best, obj)
                zero_point_best[update_mask] = zero_point[update_mask]
                scale_best[update_mask] = scale_ex[update_mask]

            scale, zero_point = scale_best, zero_point_best
            aux_loss = obj_best.sum().item()

        elif self.quant_args.param_init_method_diag == PARAM_INIT_METHODS.best_zp.value:
            zero_point, aux_loss = find_zp_matrix_fast(w / scale, 2 ** self.nbits, H_diag, norm_weight=False)
            aux_loss = aux_loss.mul(scale.square()).sum().item()
        elif self.quant_args.param_init_method_diag == PARAM_INIT_METHODS.minmax_plus.value:
            info_once("minmax plus")
            scale = (wmax - wmin) / (self.max + 1)
            scale[scale < 1e-9] = 1
            zero_point = torch.round(wmin / scale + 0.5)
            aux_loss = 0
        elif self.quant_args.param_init_method_diag == PARAM_INIT_METHODS.clip_wo_shift_zp.value:
            info_once("clip_wo_shift_zp")
            zero_point = torch.round(wmin / scale)
            aux_loss = 0
        elif self.quant_args.param_init_method_diag is None:
            info_once("minmax")
            beta = self.quant_args.scale_clip_beta
            delta = (1 - beta) * (2 ** self.nbits - 1) / (beta * 2)
            zero_point = torch.round(wmin / scale + delta)
            aux_loss = 0
        else:
            raise NotImplementedError

        self.scale = scale.float().view(-1, num_groups, 1)
        self.zero_point = zero_point.float().view(-1, num_groups, 1)
        self.num_groups = num_groups
        self.group_size = group_size

        return aux_loss

    def quant(self, w: Tensor, idx: int | Literal["all"] = "all", random_round: bool = False):
        if idx == "all":
            w = w.view(-1, self.num_groups, self.group_size)
            scale, zero_point = self.scale, self.zero_point
        else:
            scale, zero_point = self.scale[:, idx], self.zero_point[:, idx]
        if random_round:
            w_q = quant_random(w, torch.rand_like(w), scale, zero_point, self.min, self.max)
        else:
            w_q = quant(w, scale, zero_point, self.min, self.max)
        if idx == "all":
            w_q = w_q.view(-1, self.num_groups * self.group_size)
        return w_q

    def quant_int(self, w: Tensor, idx: int | Literal["all"] = "all"):
        assert idx == "all"
        w = w.view(-1, self.num_groups, self.group_size)
        w_q = quant_int(w, self.scale, self.zero_point, self.min, self.max)
        w_q = w_q.view(-1, self.num_groups * self.group_size)
        return w_q

    def dquant_int(self, w: Tensor, idx: int | Literal["all"] = "all"):
        assert idx == "all"
        w = w.view(-1, self.num_groups, self.group_size)
        w_q = dquant_int(w, self.scale, self.zero_point)
        w_q = w_q.view(-1, self.num_groups * self.group_size)
        return w_q

@torch.compile()
def quant_loss(W, W_hat, H):
    """
    W: (Out, In)
    H: (In, In)
    """
    # return torch.trace((W_hat - W) @ H @ (W_hat - W).T)
    Dt = W_hat - W
    return torch.einsum("ij,jk,ik->", Dt, H, Dt)

@torch.compile()
def quant_diag_loss(W, W_hat, H):
    Dt = W_hat - W
    return (Dt ** 2 * H.diag()).sum()

class QuantMethod(ABC):

    def __init__(self, quant_args: QuantArgs, H: torch.Tensor = None, dtype: torch.dtype = torch.float, percdamp=.01):
        self.quantizer = UniformQuantizer(quant_args)
        self.H = H
        self.quant_args = quant_args
        self.dtype = dtype

    @abstractmethod
    def quant(self, _W, **kwargs) -> Tensor:
        pass

    @torch.no_grad()
    def __call__(self, _W: Tensor):
        loss = {}
        W = _W.cuda().to(self.dtype)

        if self.quant_args.enable_H_diag_weight:
            diag_weight = self.H.diag().view(1, -1)
        else:
            diag_weight = None

        W_processed = W
        if self.quant_args.enable_magr:
            from magr import W_proximal_preprocess_groupwise
            W_processed = W_proximal_preprocess_groupwise(
                self.H, W_processed, 
                group_size=self.quant_args.group_size,
                alpha=self.quant_args.magr_alpha
            )
        loss["aux_quant_loss"] = self.quantizer.init_param(W_processed, diag_weight)

        W_hat = self.quant(W_processed)
        loss["quant_loss"] = quant_loss(W, W_hat, self.H).item()

        loss["quant_diag_loss"] = quant_diag_loss(W, W_hat, self.H).item()

        quantized_linear_param = {
            "scale": self.quantizer.scale.squeeze(-1),
            "zero": self.quantizer.zero_point.squeeze(-1).neg(),
            "qweight": pack(self.quantizer.quant_int(W_hat).clamp(0, 2 ** self.quant_args.nbits - 1).to(torch.uint8), self.quant_args.nbits)
        }

        return quantized_linear_param, loss

class RTN(QuantMethod):

    def __init__(self, quant_args, H = None, dtype = torch.float, percdamp=0.01):
        super().__init__(quant_args, H, dtype, percdamp)

        regluarization = percdamp * self.H.diagonal().mean()
        self.H.diagonal().add_(regluarization)

    def quant(self, _W: torch.Tensor):
        """
        (Out, In)
        """
        W_hat = self.quantizer.quant(_W)
        return W_hat

# @torch.compile()
def LDLQ_quant(
    W: Tensor,
    L: Tensor,
    enable_H_reorder: bool,
    quantizer: UniformQuantizer,
    idx: list[int],
    round_method: str,
    perm: Tensor = None,
    invperm: Tensor = None
):
    if enable_H_reorder:
        W = W[:, perm]
    W = W.T.contiguous()
    W_hat = torch.empty_like(W)
    B = 32
    Dt = torch.zeros((B, W.size(1)), dtype=W.dtype, device=W.device)
    for s in range(0, W.size(0), B):
        e = min(s + B, W.size(0))
        torch.mm(L[s: e, :s], W[:s] - W_hat[:s], out=Dt[:e - s])
        for i in range(s, e):
            W_hat[i] = quantizer.quant((W[i] +
                                        Dt[i - s] +
                                        L[i, s : i] @ (W[s : i] - W_hat[s : i])).view(-1, 1), idx[i], round_method == "random").view(-1)

    if enable_H_reorder:
        W_hat = W_hat[invperm]

    return W_hat.T.contiguous()


def LDLQ_quant_w_raw(
    W: Tensor,
    L: Tensor,
    enable_H_reorder: bool,
    quantizer: UniformQuantizer,
    idx: list[int],
    round_method: str,
    perm: Tensor = None,
    invperm: Tensor = None
):
    if enable_H_reorder:
        W = W[:, perm]
    W = W.T.contiguous()
    W_tilde = W.clone()
    W_hat = torch.empty_like(W)
    B = 32
    for s in range(0, W.size(0), B):
        e = min(s + B, W.size(0))
        W_tilde[s: e] += L[s: e, :s] @ (W[:s] - W_hat[:s])
        for i in range(s, e):
            W_tilde[i] += L[i, s : i] @ (W[s : i] - W_hat[s : i])
            W_hat[i] = quantizer.quant(W_tilde[i].view(-1, 1), idx[i], round_method == "random").view(-1)

    if enable_H_reorder:
        W_hat = W_hat[invperm]
        W_tilde = W_tilde[invperm]

    return W_hat.T.contiguous(), W_tilde.T.contiguous()

# @torch.compile()
def LDL_decompose(H: Tensor, dtype: torch.dtype = torch.float):
    U = cast(Tensor, torch.linalg.cholesky(H.flip([0, 1]))).flip([0, 1])
    D = U.diag()
    U.div_(D).fill_diagonal_(0)
    D = D ** 2
    return U.T.to(dtype).contiguous(), D.to(dtype)

# @torch.compile()
def GPTQ_decompose(H: torch.Tensor, dtype: torch.dtype = torch.float):
    L = torch.linalg.cholesky(H)
    L = torch.cholesky_inverse(L)
    L = cast(Tensor, torch.linalg.cholesky(L))
    D = L.diag()
    L.div_(D[None, :]).fill_diagonal_(0)
    D = 1 / D ** 2
    return L.to(dtype).contiguous(), D.to(dtype)

class LDLQ(QuantMethod):

    def __init__(self, quant_args: QuantArgs, H: Tensor = None, dtype: torch.dtype = torch.float, percdamp=.01):
        super().__init__(quant_args, H, dtype, percdamp)

        regluarization = percdamp * self.H.diagonal().mean()
        self.H.diagonal().add_(regluarization)

        self.enable_H_reorder = quant_args.enable_H_reorder
        in_dim = H.size(0)
        group_size = quant_args.group_size if quant_args.group_size > -1 else in_dim

        self.perm = self.H.diag().argsort(descending=True)
        self.invperm = self.perm.argsort()
        self.H_reorder = self.H[self.perm][:, self.perm]
        self.idx = (self.perm // group_size).tolist()

        for iter in range(1, 10 + 1):
            try:
                self.L, self.D_reorder = LDL_decompose(self.H_reorder, self.dtype)
                break
            except torch._C._LinAlgError as e:
                print("iter", iter, repr(e), flush=True)
                self.H.diagonal().add_(regluarization)
                self.H_reorder.diagonal().add_(regluarization)

        self.D = self.D_reorder[self.invperm]

    def quant(self, W: Tensor, return_W_tilde: bool = False):
        if return_W_tilde:
            return LDLQ_quant_w_raw(W, self.L, self.enable_H_reorder, self.quantizer, self.idx, self.quant_args.round_method, self.perm, self.invperm)
        return LDLQ_quant(W, self.L, self.enable_H_reorder, self.quantizer, self.idx, self.quant_args.round_method, self.perm, self.invperm)


class GPTQ(QuantMethod):

    def __init__(self, quant_args: QuantArgs, H: torch.Tensor = None, dtype: torch.dtype = torch.float, percdamp=.01):
        super().__init__(quant_args, H, dtype, percdamp)

        regluarization = percdamp * self.H.diagonal().mean()
        self.H.diagonal().add_(regluarization)

        self.enable_H_reorder = quant_args.enable_H_reorder
        in_dim = H.size(0)
        group_size = quant_args.group_size if quant_args.group_size > -1 else in_dim

        self.perm = self.H.diag().argsort(descending=True)
        self.invperm = self.perm.argsort()
        self.H_reorder = self.H[self.perm][:, self.perm]
        self.idx = (self.perm // group_size).tolist()

        for iter in range(1, 10 + 1):
            try:
                self.U, self.D_reorder = GPTQ_decompose(self.H_reorder, self.dtype)
                break
            except torch._C._LinAlgError as e:
                print("iter", iter, repr(e), flush=True)
                self.H.diagonal().add_(regluarization)
                self.H_reorder.diagonal().add_(regluarization)

        self.D = self.D_reorder[self.invperm]

    def quant(self, W: torch.Tensor):
        if self.enable_H_reorder:
            W = W[:, self.perm]
        W = W.T.contiguous()
        W_hat = torch.empty_like(W)
        B = 32
        for s in range(0, W.size(0), B):
            e = min(s + B, W.size(0))
            for i in range(s, e):
                W_hat[i] = self.quantizer.quant(W[i].view(-1, 1)).view(-1)
                W[i+1: e] += torch.outer(self.U[i+1: e, i], W_hat[i] - W[i])
            W[e:] += self.U[e:, s: e] @ (W_hat[s: e] - W[s: e])

        if self.enable_H_reorder:
            W_hat = W_hat[self.invperm]

        return W_hat.T.contiguous()


QuantMethodMap: dict[str, QuantMethod] = {
    "RTN": RTN,
    "GPTQ": GPTQ,
    "LDLQ": LDLQ,
}
