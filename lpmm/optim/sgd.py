import torch
from torch import Tensor
from typing import List, Optional

from .optimizer import LowBitOptimizer
from ..functional import vectorwise_dequant, vectorwise_quant

__all__ = ['SGD']


class SGD(LowBitOptimizer):
    def __init__(
            self, 
            params, 
            lr,
            momentum=0, 
            dampening=0,
            weight_decay=0, 
            nesterov=False, 
            qconfig=None,
            *, 
            maximize: bool = False,
        ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
            maximize=maximize,
        )
        super().__init__(params, defaults, qconfig)

    def __setstate__(self, state):
        super().__setstate__(state)

    def get_subqconfig(self, optimizer_state_name):
        if optimizer_state_name == 'momentum_buffer':
            return self.qconfig.QUANT.M
        else:
            raise ValueError(
                f""
            )

    def _init_group(
        self,
        group, 
        params_with_grad, 
        grads, 
        momentum_buffer_list,
        momentum_buffer_q_enabled,
        momentum_buffer_q_overhead,
        momentum_buffer_qmap,
    ):
        for p in group['params']:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("SGD does not support sparse gradients")
            grads.append(p.grad)
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros((), dtype=torch.float, device=p.device)
                self.init_qstate(p, "momentum_buffer")

            momentum_buffer_list.append(state['momentum_buffer'])
            momentum_buffer_q_enabled.append(self.override_q_enable[id(p)] if id(p) in self.override_q_enable else state["momentum_buffer_qstate"]["enable"])
            momentum_buffer_q_overhead.append(state["momentum_buffer_qstate"]["overhead"])
            momentum_buffer_qmap.append(state["momentum_buffer_qstate"]["qmap"])

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
            momentum_buffer_list = []
            momentum_buffer_q_enabled = []
            momentum_buffer_q_overhead = []
            momentum_buffer_qmap = []

            self._init_group(
                group, 
                params_with_grad, 
                grads,
                momentum_buffer_list,
                momentum_buffer_q_enabled,
                momentum_buffer_q_overhead,
                momentum_buffer_qmap,
            )

            kwargs = dict(
                params_with_grad=params_with_grad,
                grads=grads,
                momentum_buffer_list=momentum_buffer_list,
                momentum_buffer_q_enabled=momentum_buffer_q_enabled,
                momentum_buffer_q_overhead=momentum_buffer_q_overhead,
                momentum_buffer_qmap=momentum_buffer_qmap,
                momentum_buffer_qmetadata=self.get_qmetadata_by_state_name("momentum_buffer"),
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
            )
            _single_tensor_sgd4bit(**kwargs)

        return loss


def _single_tensor_sgd4bit(
        params_with_grad: List[Tensor],
        grads: List[Tensor],
        momentum_buffer_list: List[Tensor],
        momentum_buffer_q_enabled: List[bool],
        momentum_buffer_q_overhead: List,
        momentum_buffer_qmap: List,
        momentum_buffer_qmetadata,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
):

    for i, param in enumerate(params_with_grad):
        d_p = grads[i] if not maximize else -grads[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            # dequantize and decay
            q_overhead = momentum_buffer_q_overhead[i]
            q_buf = momentum_buffer_list[i]
            if q_buf.numel() <= 1:
                # decay not needed when initializing first moment 
                q_buf.data = buf = torch.clone(d_p).detach()
            else:
                if momentum_buffer_q_enabled[i]:
                    q_overhead.update(momentum_buffer_qmetadata)
                    buf = vectorwise_dequant(q_buf, qmap=momentum_buffer_qmap[i], shape=param.shape, **q_overhead)
                    q_overhead.clear()
                else:
                    buf = q_buf
                
                # decay
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            # udpate
            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)

        # quantize
        if momentum != 0:
            if momentum_buffer_q_enabled[i]:
                qx, gen = vectorwise_quant(buf, qmap=momentum_buffer_qmap[i], shape=param.shape, **momentum_buffer_qmetadata)
                q_buf.data = qx
                q_overhead.update(gen)
