import torch
import math

from torch import Tensor
from typing import List, Optional
import time

from .optimizer import LowBitOptimizer
from ..functional import vectorwise_dequant, vectorwise_quant

__all__ = ["AdamW"]


class AdamW(LowBitOptimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        use_first_moment=True,
        factor_second_moment=False,
        qconfig=None,
        *,
        fused: Optional[bool] = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=fused,
            use_first_moment=use_first_moment,
            factor_second_moment=factor_second_moment,
        )
        super().__init__(params, defaults, qconfig)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("fused", None)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def get_subqconfig(self, optimizer_state_name):
        if optimizer_state_name == 'exp_avg':
            return self.qconfig.QUANT.M
        elif optimizer_state_name == 'exp_avg_sq':
            return self.qconfig.QUANT.SQM
        else:
            raise ValueError(
                f""
            )

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2 and param_group["factor_second_moment"]
        use_first_moment = param_group["use_first_moment"]
        return factored, use_first_moment
    
    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        # copy from fairseq's adafactor implementation:
        # https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2)
        return torch.mul(r_factor, c_factor)
    
    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        exp_avg_sqs_factored,
        exp_avg_sq_rows,
        exp_avg_sq_cols,
        state_steps,
        exp_avgs_q_enabled,
        exp_avg_sqs_q_enabled,
        exp_avgs_q_overhead,
        exp_avg_sqs_q_overhead,
        exp_avgs_qmap,
        exp_avg_sqs_qmap,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            # if p.grad.dtype in {torch.float16, torch.bfloat16}:
            #     p.grad = p.grad.float()
            grads.append(p.grad)
            state = self.state[p]

            factored, _ = self._get_options(group, p.shape)
            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = torch.tensor(0.0)
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros((), dtype=torch.float, device=p.device)
                self.init_qstate(p, "exp_avg")
                # Exponential moving average of squared gradient values
                if factored:
                    state["exp_avg_sq_row"] = torch.zeros(p.shape[:-1], device=p.device)
                    state["exp_avg_sq_col"] = torch.zeros(p.shape[:-2] + p.shape[-1:], device=p.device)
                else:
                    state["exp_avg_sq"] = torch.zeros((), dtype=torch.float, device=p.device)
                self.init_qstate(p, "exp_avg_sq")

            state_steps.append(state["step"])
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs_factored.append(factored)
            if factored:
                exp_avg_sq_rows.append(state["exp_avg_sq_row"])
                exp_avg_sq_cols.append(state["exp_avg_sq_col"])
                exp_avg_sqs.append(None)
            else:
                exp_avg_sq_rows.append(None)
                exp_avg_sq_cols.append(None)
                exp_avg_sqs.append(state["exp_avg_sq"])

            exp_avgs_q_enabled.append(self.override_q_enable[id(p)] if id(p) in self.override_q_enable else state["exp_avg_qstate"]["enable"])
            exp_avg_sqs_q_enabled.append(self.override_q_enable[id(p)] if id(p) in self.override_q_enable else state["exp_avg_sq_qstate"]["enable"])
            exp_avgs_q_overhead.append(state["exp_avg_qstate"]["overhead"])
            exp_avg_sqs_q_overhead.append(state["exp_avg_sq_qstate"]["overhead"])
            exp_avgs_qmap.append(state["exp_avg_qstate"]["qmap"])
            exp_avg_sqs_qmap.append(state["exp_avg_sq_qstate"]["qmap"])


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avg_sqs_factored = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_sq_rows = []
            exp_avg_sq_cols = []
            state_steps = []
            beta1, beta2 = group["betas"]
            exp_avgs_q_enabled = []
            exp_avg_sqs_q_enabled = []
            exp_avgs_q_overhead = []
            exp_avg_sqs_q_overhead = []
            exp_avgs_qmap = []
            exp_avg_sqs_qmap = []

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                exp_avg_sqs_factored,
                exp_avg_sq_rows,
                exp_avg_sq_cols,
                state_steps,
                exp_avgs_q_enabled,
                exp_avg_sqs_q_enabled,
                exp_avgs_q_overhead,
                exp_avg_sqs_q_overhead,
                exp_avgs_qmap,
                exp_avg_sqs_qmap,
            )

            kwargs = dict(
                params_with_grad=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                exp_avg_sqs_factored=exp_avg_sqs_factored,
                exp_avg_sq_rows=exp_avg_sq_rows,
                exp_avg_sq_cols=exp_avg_sq_cols,
                state_steps=state_steps,
                exp_avgs_q_enabled=exp_avgs_q_enabled,
                exp_avg_sqs_q_enabled=exp_avg_sqs_q_enabled,
                exp_avgs_q_overhead=exp_avgs_q_overhead,
                exp_avg_sqs_q_overhead=exp_avg_sqs_q_overhead,
                exp_avgs_qmap=exp_avgs_qmap,
                exp_avg_sqs_qmap=exp_avg_sqs_qmap,
                exp_avg_qmetadata=self.get_qmetadata_by_state_name("exp_avg"),
                exp_avg_sq_qmetadata=self.get_qmetadata_by_state_name("exp_avg_sq"),
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )

            if group["fused"] and torch.jit.is_scripting():
                raise RuntimeError("torch.jit.script not supported with fused optimizers")

            if group["fused"] and not torch.jit.is_scripting():
                _fused_adamw4bit(**kwargs)
            else:
                _single_tensor_adamw4bit(**kwargs)

            # beta1, beta2 = group["betas"]
            # lr = group["lr"]
            # weight_decay = group["weight_decay"]
            # eps = group["eps"]

            # for p in group["params"]:
            #     if p.grad is None:
            #         continue
            #     grad = p.grad.data
            #     if grad.dtype in {torch.float16, torch.bfloat16}:
            #         grad = grad.float()
            #     if p.grad.is_sparse:
            #         raise RuntimeError("AdamW does not support sparse gradients")

            #     state = self.state[p]
            #     grad_shape = p.grad.shape

            #     factored, use_first_moment = self._get_options(group, grad_shape)
            #     # State initialization
            #     if len(state) == 0:
            #         # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
            #         # This is because kernel launches are costly on CUDA and XLA.
            #         state["step"] = 0
            #         # Exponential moving average of gradient values
            #         if use_first_moment:
            #             state["exp_avg"] = torch.tensor(0.0)
            #         # Exponential moving average of squared gradient values
            #         if factored:
            #             state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
            #             state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
            #         else:
            #             state["exp_avg_sq"] = torch.tensor(0.0)
            #         # quantization state
            #         self.init_qstate(p)

            #     # take out optimizer state
            #     param = p
            #     # dequantize
            #     if use_first_moment:
            #         exp_avg = state["exp_avg"]
            #         if exp_avg.numel() <= 1:
            #             exp_avg.data = torch.zeros_like(param, memory_format=torch.preserve_format)
            #         else:
            #             hat_exp_avg = self.dequantize(param, 'exp_avg', exp_avg)
            #             if hat_exp_avg is not None:
            #                 exp_avg.data = hat_exp_avg
            #             del hat_exp_avg
            #     else:
            #         exp_avg = grad
            #     if factored:
            #         exp_avg_sq_row = state["exp_avg_sq_row"]
            #         exp_avg_sq_col = state["exp_avg_sq_col"]
            #     else:
            #         exp_avg_sq = state["exp_avg_sq"]
            #         if exp_avg_sq.numel() <= 1:
            #             exp_avg_sq.data = torch.zeros_like(param, memory_format=torch.preserve_format)
            #         else:
            #             hat_exp_avg_sq = self.dequantize(param, 'exp_avg_sq', exp_avg_sq)
            #             if hat_exp_avg_sq is not None:
            #                 exp_avg_sq.data = hat_exp_avg_sq
            #             del hat_exp_avg_sq

            #     # update
            #     state["step"] += 1
            #     # Perform stepweight decay
            #     param.mul_(1 - lr * weight_decay)

            #     # Decay the first and second moment running average coefficient
            #     if use_first_moment:
            #         exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            #     if factored:
            #         update = (grad ** 2)
            #         exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1 - beta2)
            #         exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1 - beta2)
            #         exp_avg_sq = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
            #     else:
            #         exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            #     step = state["step"]
            #     bias_correction1 = 1 - beta1 ** step
            #     bias_correction2 = 1 - beta2 ** step
            #     step_size = lr / bias_correction1
            #     bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

            #     denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            #     param.addcdiv_(exp_avg, denom, value=-step_size)

            #     # take in optimizer state
            #     if use_first_moment:
            #         q_exp_avg = self.quantize(param, 'exp_avg', exp_avg)
            #         if q_exp_avg is not None:
            #             exp_avg.data = q_exp_avg
            #     if not factored:
            #         q_exp_avg_sq = self.quantize(param, 'exp_avg_sq', exp_avg_sq)
            #         if q_exp_avg_sq is not None:
            #             exp_avg_sq.data = q_exp_avg_sq

        return loss


def _single_tensor_adamw4bit(
    params_with_grad: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_sqs_factored: List[bool],
    exp_avg_sq_rows: List[Tensor],
    exp_avg_sq_cols: List[Tensor],
    state_steps: List[Tensor],
    exp_avgs_q_enabled: List[bool],
    exp_avg_sqs_q_enabled: List[bool],
    exp_avgs_q_overhead: List,
    exp_avg_sqs_q_overhead: List,
    exp_avgs_qmap: List,
    exp_avg_sqs_qmap: List,
    exp_avg_qmetadata,
    exp_avg_sq_qmetadata,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float
):

    for i, param in enumerate(params_with_grad):
        grad = grads[i]
        q_exp_avg = exp_avgs[i]
        q_exp_avg_sq = exp_avg_sqs[i]
        exp_avg_sq_row = exp_avg_sq_rows[i]
        exp_avg_sq_col = exp_avg_sq_cols[i]
        factored = exp_avg_sqs_factored[i]
        step_t = state_steps[i]

        # update step
        step_t += 1
        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        if factored:
            _single_quantized_factored_update(
                param,
                grad,
                q_exp_avg,
                exp_avg_sq_row,
                exp_avg_sq_col,
                exp_avgs_q_enabled[i],
                exp_avgs_q_overhead[i],
                exp_avgs_qmap[i],
                exp_avg_qmetadata,
                lr,
                beta1,
                beta2,
                eps,
                step_t.item()
            )

        else:
            exp_avg_q_overhead = exp_avgs_q_overhead[i]
            exp_avg_sq_q_overhead = exp_avg_sqs_q_overhead[i]

            # dequantize
            if q_exp_avg.numel() <= 1:
                q_exp_avg.data = exp_avg = torch.zeros_like(param, memory_format=torch.preserve_format)
            elif exp_avgs_q_enabled[i]:
                exp_avg_q_overhead.update(exp_avg_qmetadata)
                exp_avg = vectorwise_dequant(q_exp_avg, qmap=exp_avgs_qmap[i], shape=param.shape, **exp_avg_q_overhead)
                exp_avg_q_overhead.clear()
            else:
                exp_avg = q_exp_avg
            if q_exp_avg_sq.numel() <= 1:
                q_exp_avg_sq.data = exp_avg_sq = torch.zeros_like(param, memory_format=torch.preserve_format)
            elif exp_avg_sqs_q_enabled[i]:
                exp_avg_sq_q_overhead.update(exp_avg_sq_qmetadata)
                exp_avg_sq = vectorwise_dequant(q_exp_avg_sq, qmap=exp_avg_sqs_qmap[i], shape=param.shape, **exp_avg_sq_q_overhead)
                exp_avg_sq_q_overhead.clear()
            else:
                exp_avg_sq = q_exp_avg_sq

            # Decay the first and second moment running average coefficient
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            step = step_t.item()
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)

            # quantize
            if exp_avgs_q_enabled[i]:
                qx, gen = vectorwise_quant(exp_avg, qmap=exp_avgs_qmap[i], shape=param.shape, **exp_avg_qmetadata)
                q_exp_avg.data = qx
                exp_avg_q_overhead.update(gen)
            else:
                pass
            if exp_avg_sqs_q_enabled[i]:
                qx, gen = vectorwise_quant(exp_avg_sq, qmap=exp_avg_sqs_qmap[i], shape=param.shape, **exp_avg_sq_qmetadata)
                q_exp_avg_sq.data = qx
                exp_avg_sq_q_overhead.update(gen)
            else:
                pass
            

def _fused_adamw4bit(
    params_with_grad: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_sqs_factored: List[bool],
    exp_avg_sq_rows: List[Tensor],
    exp_avg_sq_cols: List[Tensor],
    state_steps: List[Tensor],
    exp_avgs_q_enabled: List[bool],
    exp_avg_sqs_q_enabled: List[bool],
    exp_avgs_q_overhead: List,
    exp_avg_sqs_q_overhead: List,
    exp_avgs_qmap: List,
    exp_avg_sqs_qmap: List,
    exp_avg_qmetadata,
    exp_avg_sq_qmetadata,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float
):
    for i, param in enumerate(params_with_grad):
        grad = grads[i]
        q_exp_avg = exp_avgs[i]
        q_exp_avg_sq = exp_avg_sqs[i]
        exp_avg_sq_row = exp_avg_sq_rows[i]
        exp_avg_sq_col = exp_avg_sq_cols[i]
        factored = exp_avg_sqs_factored[i]
        step_t = state_steps[i]

        if factored:
            # fused_adam4bit do not apply to factored case
            
            # update step
            step_t += 1
            # Perform stepweight decay
            param.mul_(1 - lr * weight_decay)

            _single_quantized_factored_update(
                param,
                grad,
                q_exp_avg,
                exp_avg_sq_row,
                exp_avg_sq_col,
                exp_avgs_q_enabled[i],
                exp_avgs_q_overhead[i],
                exp_avgs_qmap[i],
                exp_avg_qmetadata,
                lr,
                beta1,
                beta2,
                eps,
                step_t.item()
            )
        else:
            # update step
            step_t += 1
            if exp_avgs_q_enabled[i] != exp_avg_sqs_q_enabled[i]:
                raise ValueError(f"For same tensor, exp_avg and exp_avg_sq should be both quantized or unquantized simultaneously,"
                                 f" but get ({exp_avgs_q_enabled[i]} {exp_avg_sqs_q_enabled[i]})")
            if exp_avgs_q_enabled[i]:
                if exp_avg_qmetadata["scale_type"] != "group":
                    print(f"Warning: fused_adamw4bit only support block-wise scaling, but get exp_avg scale_type {exp_avg_qmetadata['scale_type']}.")
                if exp_avg_sq_qmetadata["scale_type"] != "group":
                    print(f"Warning: fused_adamw4bit only support block-wise scaling, but get exp_avg_sq scale_type {exp_avg_sq_qmetadata['scale_type']}.")

                bytelength = (param.numel() + 1) // 2
                if q_exp_avg.numel() <= 1:
                    q_exp_avg.data = torch.zeros((bytelength,), dtype=torch.int8, device=param.device)
                if q_exp_avg_sq.numel() <= 1:
                    q_exp_avg_sq.data = torch.zeros((bytelength,), dtype=torch.int8, device=param.device)
                blocks = (param.numel() + 127) // 128
                if "max1" in exp_avgs_q_overhead[i]:
                    exp_avg_scale = exp_avgs_q_overhead[i]["max1"]
                else:
                    exp_avg_scale = torch.zeros((blocks,), dtype=torch.float32, device=param.device)
                    exp_avgs_q_overhead[i]["max1"] = exp_avg_scale
                if "max1" in exp_avg_sqs_q_overhead[i]:
                    exp_avg_sq_scale = exp_avg_sqs_q_overhead[i]["max1"]
                else:
                    exp_avg_sq_scale = torch.zeros((blocks,), dtype=torch.float32, device=param.device)
                    exp_avg_sqs_q_overhead[i]["max1"] = exp_avg_sq_scale

                with torch.cuda.device(param.device):
                    import lpmm.cpp_extension.fused_adamw as fused_adamw
                    fused_adamw.adamw4bit_single_tensor(
                        param,
                        grad,
                        q_exp_avg,
                        q_exp_avg_sq,
                        exp_avg_scale,
                        exp_avg_sq_scale,
                        exp_avgs_qmap[i],
                        exp_avg_sqs_qmap[i],
                        beta1,
                        beta2,
                        lr,
                        weight_decay,
                        eps,
                        step_t.item(),
                    )
            else:
                if q_exp_avg.numel() <= 1:
                    q_exp_avg.data = torch.zeros_like(param, memory_format=torch.preserve_format)
                if q_exp_avg_sq.numel() <= 1:
                    q_exp_avg_sq.data = torch.zeros_like(param, memory_format=torch.preserve_format)
                with torch.cuda.device(param.device):
                    import lpmm.cpp_extension.fused_adamw as fused_adamw
                    fused_adamw.adamw_single_tensor(
                        param,
                        grad,
                        q_exp_avg,
                        q_exp_avg_sq,
                        beta1,
                        beta2,
                        lr,
                        weight_decay,
                        eps,
                        step_t.item(),
                    )


def _dispatch_sqrt(x: float):  # float annotation is needed because of torchscript type inference
    if not torch.jit.is_scripting() and isinstance(x, torch.Tensor):
        return x.sqrt()
    else:
        return math.sqrt(x)


def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        # copy from fairseq's adafactor implementation:
        # https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2)
        return torch.mul(r_factor, c_factor)
    

def _single_quantized_factored_update(
    param,
    grad,
    q_exp_avg,
    exp_avg_sq_row,
    exp_avg_sq_col,
    exp_avg_q_enabled,
    exp_avg_q_overhead,
    exp_avg_qmap,
    exp_avg_qmetadata,
    lr,
    beta1,
    beta2,
    eps,
    step,
):
    # dequantize
    if q_exp_avg.numel() <= 1:
        q_exp_avg.data = exp_avg = torch.zeros_like(param, memory_format=torch.preserve_format)
    elif exp_avg_q_enabled:
        exp_avg_q_overhead = exp_avg_q_overhead
        exp_avg_q_overhead.update(exp_avg_qmetadata)
        exp_avg = vectorwise_dequant(q_exp_avg, qmap=exp_avg_qmap, shape=param.shape, **exp_avg_q_overhead)
        exp_avg_q_overhead.clear()
    else:
        exp_avg = q_exp_avg

    # Decay the first and second moment running average coefficient
    exp_avg.lerp_(grad, 1 - beta1)
    update = (grad ** 2)
    exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1 - beta2)
    exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1 - beta2)
    exp_avg_sq = _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    step_size = lr / bias_correction1
    bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
    param.addcdiv_(exp_avg, denom, value=-step_size)

    # quantize
    if exp_avg_q_enabled:
        qx, gen = vectorwise_quant(exp_avg, qmap=exp_avg_qmap, shape=param.shape, **exp_avg_qmetadata)
        q_exp_avg.data = qx
        exp_avg_q_overhead.update(gen)
    else:
        pass
