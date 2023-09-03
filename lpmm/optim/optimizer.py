from collections import abc as container_abcs
from collections import defaultdict
from copy import deepcopy
from itertools import chain

import torch

from ..functional import create_general_qmap, init_lpmm_generator
from ..utils import get_rank
from ..config import get_config

compression_time = 0

class LowBitOptimizer(torch.optim.Optimizer):
    def __init__(self, params, defaults, config):
        super(LowBitOptimizer, self).__init__(params, defaults)

        # init lpmm generator
        if torch.distributed.is_initialized():
            seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
            torch.distributed.broadcast(seed, src=0)
            init_lpmm_generator(get_rank(), seed.item()) # no stochastic rounding
        
        self.qconfig = get_config(config)
        self.override_q_enable = {}
        self.qmaps = {}

    def override_quantize_enable(self, module, param_name, enable):
        p = getattr(module, param_name)
        assert p is not None
        assert isinstance(p, torch.Tensor) or isinstance(p, torch.Parameter)
        if len(self.state[p]) != 0:
            raise ValueError("overriding enabling of quantized parameters is prohibited after state initialization.")
        self.override_q_enable[id(p)] = enable

    def init_qstate(self, p, state_name):
        state = self.state[p]
        field = f"{state_name}_qstate"
        state[field] = {
            "enable": True, 
            "overhead": dict(),
            "qmap": None,
        }
        subconfig = self.get_subqconfig(state_name)
        state[field][
            "enable"
        ] = _get_qenable_fn(p, subconfig.ENABLE, subconfig.THRESHOLD)
        
        md = self.get_qmetadata_by_state_name(state_name)
        qmap_key = (md['quant_type'], md['b'], md['signed'])
        if qmap_key not in self.qmaps:
            self.qmaps[qmap_key] = create_general_qmap(*qmap_key)
        self.qmaps[qmap_key] = self.qmaps[qmap_key].to(p.device)
        state[field]["qmap"] = self.qmaps[qmap_key]

    def get_qmetadata_by_state_name(self, optimizer_state_name):
        subconfig = self.get_subqconfig(optimizer_state_name)
        md = dict(
            b=subconfig.BITS,
            scale_type=subconfig.SCALE_TYPE.DEFAULT,
            quant_type=subconfig.QUANT_TYPE.DEFAULT,
            round_type=subconfig.ROUND_TYPE,
            gp_sz=subconfig.GROUP_SIZE,
            signed=subconfig.SIGNED,
        )
        return md

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['qconfig'] = self.qconfig
        return state_dict
    
    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.qconfig = state_dict['qconfig']

        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = dict(zip(chain.from_iterable((g['params'] for g in saved_groups)),
                      chain.from_iterable((g['params'] for g in groups))))

        def cast(param, value, key=None):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                # Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424
                if (key != "step"):
                    if param.is_floating_point() and value.dtype != torch.int8:
                        value = value.to(param.dtype)
                    value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v, key=k) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})


    @torch.no_grad()
    def step(self, closure=None):
        r"""Performs a single optimization step with quantization.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        raise NotImplementedError(
            'The step method needs overriding'
        )
    
    def get_subqconfig(self, optimizer_state_name):
        raise NotImplementedError(
            'The get_subconfig method needs overriding'
        )


def _get_qenable_fn(p, prior_enable, th):
    if not prior_enable:
        return False
    # if p.dim() < 2:
    #     return False
    if th is not None and p.numel() <= th:
        return False
    return True