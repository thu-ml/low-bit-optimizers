import torch
import os
import yaml
from yacs.config import CfgNode as CN
from .utils import get_rank, get_world_size

_C = CN()

# Base config files
_C.BASE = ['']

_C.QUANT = CN(new_allowed=True)
_C.QUANT.INIT_STATES = ['param']

_C.QUANT.P = CN(new_allowed=True)
_C.QUANT.P.ENABLE = False
_C.QUANT.P.THRESHOLD = 4096
_C.QUANT.P.EXCLUDE_SUFFIX = [''] # model-related
_C.QUANT.P.EXCLUDE_REGEX = ['']
_C.QUANT.P.BITS = 8
_C.QUANT.P.SCALE_TYPE = CN(new_allowed=True)
_C.QUANT.P.SCALE_TYPE.DEFAULT = 'group'
_C.QUANT.P.SCALE_TYPE.DEFAULT_ONLY = True
_C.QUANT.P.QUANT_TYPE = CN(new_allowed=True)
_C.QUANT.P.QUANT_TYPE.DEFAULT = 'linear'
_C.QUANT.P.QUANT_TYPE.DEFAULT_ONLY = True
_C.QUANT.P.ROUND_TYPE = 'sr'
_C.QUANT.P.GROUP_SIZE = 64
_C.QUANT.P.SIGNED = True

_C.QUANT.G = CN(new_allowed=True)
_C.QUANT.G.ENABLE = False
_C.QUANT.G.THRESHOLD = 4096

_C.QUANT.M = CN(new_allowed=True)
_C.QUANT.M.ENABLE = True
_C.QUANT.M.THRESHOLD = 4096
_C.QUANT.M.EXCLUDE_SUFFIX = [''] # model-related
_C.QUANT.M.EXCLUDE_REGEX = ['']
_C.QUANT.M.BITS = 4
_C.QUANT.M.SCALE_TYPE = CN(new_allowed=True)
_C.QUANT.M.SCALE_TYPE.DEFAULT = 'group'
_C.QUANT.M.SCALE_TYPE.DEFAULT_ONLY = True
_C.QUANT.M.QUANT_TYPE = CN(new_allowed=True)
_C.QUANT.M.QUANT_TYPE.DEFAULT = 'nonlinear'
_C.QUANT.M.QUANT_TYPE.DEFAULT_ONLY = True
_C.QUANT.M.ROUND_TYPE = 'real-nearest'
_C.QUANT.M.GROUP_SIZE = 128
_C.QUANT.M.SIGNED = True

_C.QUANT.SQM = CN(new_allowed=True)
_C.QUANT.SQM.ENABLE = True
_C.QUANT.SQM.THRESHOLD = 4096
_C.QUANT.SQM.EXCLUDE_SUFFIX = [''] # model-related
_C.QUANT.SQM.EXCLUDE_REGEX = ['']
_C.QUANT.SQM.BITS = 4
_C.QUANT.SQM.SCALE_TYPE = CN(new_allowed=True)
_C.QUANT.SQM.SCALE_TYPE.DEFAULT = 'group'
_C.QUANT.SQM.SCALE_TYPE.DEFAULT_ONLY = True
_C.QUANT.SQM.QUANT_TYPE = CN(new_allowed=True)
_C.QUANT.SQM.QUANT_TYPE.DEFAULT = 'power-1'
_C.QUANT.SQM.QUANT_TYPE.DEFAULT_ONLY = True
_C.QUANT.SQM.ROUND_TYPE = 'real-nearest'
_C.QUANT.SQM.GROUP_SIZE = 128
_C.QUANT.SQM.SIGNED = False

_C.QUANT.DEBUG = CN(new_allowed=True)
_C.QUANT.DEBUG.TRUNCATED_RATE_STAT_ITER = False
_C.QUANT.DEBUG.ROW_ABSMAX_STAT_ITER = False
_C.QUANT.DEBUG.ROW_ABSMAX_STAT_EPOCH = False

_C.TRAIN = CN(new_allowed=True)

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.OUTPUT = '.'
_C.TAG = '' # (optional) for index of repeat experiments
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    def _check_args(name):
        if hasattr(args, name) and getattr(args, name) is not None:
            return True
        return False
    
    config.defrost()
    if _check_args('output'):
        config.OUTPUT = args.output
    elif _check_args('workspace'):
        config.OUTPUT = args.workspace
    elif _check_args('output_dir'):
        config.OUTPUT = args.output_dir
    elif _check_args('outdir'):
        config.OUTPUT = args.outdir
    elif _check_args('save_dir'):
        config.OUTPUT = args.save_dir
    elif _check_args('work_dir'):
        config.OUTPUT = args.work_dir
    if _check_args('tag'):
        config.TAG = args.tag
    # output folder, make sure that is consistent with the main output foler
    config.OUTPUT = os.path.join(config.OUTPUT, config.TAG)
    config.freeze()

    if _check_args('q_cfg'):
        if args.q_cfg is not None:
            _update_config_from_file(config, args.q_cfg)
            return

    config.defrost()
    if _check_args('lpmm_enable'):
        config.QUANT.P.ENABLE = bool(args.lpmm_enable & 1)
        config.QUANT.G.ENABLE = bool(args.lpmm_enable & 2)
        config.QUANT.M.ENABLE = bool(args.lpmm_enable & 4)
        config.QUANT.SQM.ENABLE = bool(args.lpmm_enable & 8)
    if _check_args('pb'):
        config.QUANT.P.BITS = args.pb
    if _check_args('gb'):
        config.QUANT.G.BITS = args.gb
    if _check_args('mb'):
        config.QUANT.M.BITS = args.mb
    if _check_args('sqmb'):
        config.QUANT.SQM.BITS = args.sqmb
    if _check_args('round_type'):
        if args.round_type in ['sr', 'up', 'down', 'nearest', 'sr1', 'real-nearest', 'real-sr']:
            config.QUANT.P.ROUND_TYPE = args.round_type
            config.QUANT.M.ROUND_TYPE = args.round_type
            config.QUANT.SQM.ROUND_TYPE = args.round_type
    if _check_args('scale_type'):
        if args.scale_type in ['tensor', 'dim0', 'dim1', 'dim01', 'dim10', 'group', 'rank1', 'rank1-group']:
            # config.QUANT.P.SCALE_TYPE.DEFAULT = args.scale_type
            # config.QUANT.M.SCALE_TYPE.DEFAULT = args.scale_type
            config.QUANT.SQM.SCALE_TYPE.DEFAULT = args.scale_type
        if args.scale_type[:5] == 'group' and len(args.scale_type) > 5:
            group_size = int(args.scale_type[5:]) # format 'group[xxx]' where 'xxx' is the exact group size
            # config.QUANT.P.SCALE_TYPE.DEFAULT = 'group'
            # config.QUANT.M.SCALE_TYPE.DEFAULT = 'group'
            config.QUANT.SQM.SCALE_TYPE.DEFAULT = 'group'
            config.QUANT.M.GROUP_SIZE = group_size
            config.QUANT.SQM.GROUP_SIZE = group_size
    if _check_args('q_oracle'): # NOTE: improvising
        if args.q_oracle in ['linear', 'nonlinear', 'nonlinear-nozero',
                             'power-1', 'power-2', 'power-3',
                             'float-point']:
            # config.QUANT.P.QUANT_TYPE.DEFAULT = args.q_oracle
            config.QUANT.M.QUANT_TYPE.DEFAULT = args.q_oracle
            config.QUANT.SQM.QUANT_TYPE.DEFAULT = args.q_oracle
    if _check_args('group_size'):
        if args.group_size > 0 and config.QUANT.M.SCALE_TYPE.DEFAULT == 'group':
            print(f"[Warn] Set M.GROUP_SIZE from {config.QUANT.M.GROUP_SIZE} to {args.group_size}.")
            config.QUANT.M.GROUP_SIZE = args.group_size

    # set local rank for distributed training
    if _check_args('local_rank'):
        config.LOCAL_RANK = args.local_rank

    config.freeze()

    # init output dir
    if get_rank() == 0:
        os.makedirs(config.OUTPUT, exist_ok=True)


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    if isinstance(args, str) :
        _update_config_from_file(config, args)
    elif args is not None:
        update_config(config, args)

    if get_rank() == 0:
        print(config)
        if config.OUTPUT is not None:
            config_file = os.path.join(config.OUTPUT, "lpmm_config.txt")
            with open(config_file, "w") as fout:
                fout.write(str(config))

    return config