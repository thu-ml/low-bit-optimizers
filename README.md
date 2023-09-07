# Low-bit Optimizers

Offical implementation of the paper: *[Memory Efficient Optimizers with 4-bit States](https://arxiv.org/abs/2309.01507)*.

Optimizer states are a major source of memory consumption for training neural networks, limiting the maximum trainable model within given memory budget. Compressing the optimizer states from 32-bit floating points to lower bitwidth is promising to reduce the training memory footprint, while the current lowest achievable bitwidth is 8-bit. In this work, we push optimizer states bitwidth down to 4-bit through a detailed empirical analysis of first and second order momentums. Specifically, we find that momentums have complicated outlier patterns, that current block-wise quantization cannot accurately approximate. We use a smaller block size and propose to utilize both row-wise and column-wise information for better quantization. We further identify a zero point problem of quantizing the second-order momentum, and solve this problem with a linear quantizer that excludes the zero point. Our 4-bit optimizer is evaluated on a wide variety of benchmarks including natural language understanding, machine translation, image classification, and instruction tuning. On all the tasks our optimizers can achieve comparable accuracy with their full-precision counterparts, while enjoying better memory efficiency. 

## Installation

**Requirements**
Python >= 3.7 + CUDA >= 11.0 + torch >= 1.13.0.

To install run:

```bash
git clone https://github.com/thu-ml/low-bit-optimizers.git
pip install -v -e .
```

## Usage

### Using 4-bit Optimizers

To get started with 4-bit optimizers, simply replace your existing optimizer with one of our 4-bit optimizers: 4-bit AdamW, 4-bit Factor, or 4-bit AdamW (fused).

```python
import lpmm

# Comment out or remove the old optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

# Use 4-bit AdamW
optimizer = lpmm.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

# Or, use 4-bit Factor
optimizer = lpmm.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), factor_second_moment=True)

# Or, use 4-bit AdamW (fused)
optimizer = lpmm.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), fused=True)
```

Currently, the supported optimizers are Adam (AdamW) and SGD.

### Modifying Quantization Hyperparameters

To modify the quantization configuration (e.g., normalization function, quantization map, bits, etc.) of non-fused optimizers, create a new configuration file and pass its file path to the optimizer using the `qconfig` argument. Example configurations can be found in the [lpmm/configs](lpmm/configs) directory.
By default, the quantization configuration for non-fused optimizers is specified in [lpmm/configs/default.yml](lpmm/configs/default.yml), while for fused optimizers, it is specified in [lpmm/configs/2nd_moment_group_128.yml](lpmm/configs/2nd_moment_group_128.yml). The configuration for fused optimizers is currently fixed and cannot be changed.

To use a new configuration file, follow the example below:

```python
config_path = f"configs/default.yml" # path to your configuration file
optimizer = lpmm.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), qconfig=config_path)
```
Commonly used hyperparameters and their possible values include:
- SCALE_TYPE (normalization function): tensor, dim0, dim1, group, rank1
- QUANT_TYPE (quantization map): nonlinear, power-1, power-2
- BITS: 4, 5, 6, 7, 8
- ENABLE (whether to quantize the state): True, False

We recommend to use BITS = 4 or 8.

### Overriding Quantization Enablement for Specific Parameters

To optimize certain parameters using 32-bit precision instead of quantizing them, use the `override_quantize_enable` method as shown below:

```python
optimizer = lpmm.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
optimizer.override_quantize_enable(module, param_name, enable=False)
```

In this example, `module` is the module containing the parameter, and `param_name` is the name of the parameter you wish to optimize with 32-bit precision. Setting `enable=False` will prevent quantization of the specified parameter.