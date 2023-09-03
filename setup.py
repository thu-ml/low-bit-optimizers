# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import glob
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension




setup(
    name=f"lpmm",
    description="low-bit optimizers.",
    keywords="gpu optimizers optimization low-bit quantization compression",
    url="https://github.com/thu-ml/low-bit-optimizers",
    packages=find_packages(),
    cmdclass={'build_ext': BuildExtension},
    ext_modules=[
        CUDAExtension(
            'lpmm.cpp_extension.quantization',
            ['lpmm/cpp_extension/quantization.cc', 'lpmm/cpp_extension/quantization_kernel.cu']
        ),
        CUDAExtension(
            'lpmm.cpp_extension.fused_adamw',
            ['lpmm/cpp_extension/fused_adamw.cc', 'lpmm/cpp_extension/fused_adamw_kernel.cu']
        ),
    ],
)