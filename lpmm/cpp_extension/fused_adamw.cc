#include <torch/extension.h>
#include <torch/torch.h>

#include "common.h"

using torch::Tensor;

void fused_adamw_cuda(Tensor& p, Tensor& g, Tensor& exp_avg, 
          Tensor& exp_avg_sq, float beta1, float beta2,
          float lr, float weight_decay, float eps, float step);
void fused_adamw4bit_cuda(Tensor& p, Tensor& g, Tensor& exp_avg, Tensor& exp_avg_sq, 
          Tensor& exp_avg_scale, Tensor& exp_avg_sq_scale, Tensor& exp_avg_qmap, Tensor& exp_avg_sq_qmap,
          float beta1, float beta2, float lr, float weight_decay, float eps, float step);

void adamw_single_tensor(
    Tensor& p,
    Tensor& g, 
    Tensor& exp_avg, 
    Tensor& exp_avg_sq, 
    float beta1,
    float beta2,
    float lr,
    float weight_decay,
    float eps,
    float step
) {
  CHECK_CUDA_TENSOR_FLOAT(p);
  CHECK_CUDA_TENSOR_FLOAT(exp_avg);
  CHECK_CUDA_TENSOR_FLOAT(exp_avg_sq);
  CHECK_CUDA_TENSOR_FLOAT(g);
  int64_t num_elem = p.numel();
  AT_ASSERTM(exp_avg.numel() == num_elem,
             "number of elements in exp_avg and p tensors should be equal");
  AT_ASSERTM(exp_avg_sq.numel() == num_elem,
             "number of elements in exp_avg_sq and p tensors should be equal");
  AT_ASSERTM(g.numel() == num_elem,
             "number of elements in g and p tensors should be equal");

  fused_adamw_cuda(p, g, exp_avg, exp_avg_sq,
                   beta1, beta2, lr, weight_decay, eps, step);
}


void adamw4bit_single_tensor(
    Tensor& p,
    Tensor& g, 
    Tensor& exp_avg, 
    Tensor& exp_avg_sq,
    Tensor& exp_avg_scale, 
    Tensor& exp_avg_sq_scale, 
    Tensor& exp_avg_qmap, 
    Tensor& exp_avg_sq_qmap,
    float beta1,
    float beta2,
    float lr,
    float weight_decay,
    float eps,
    float step
) {
  CHECK_CUDA_TENSOR_FLOAT(p);
  CHECK_CUDA_TENSOR_FLOAT(g);
  CHECK_CUDA_TENSOR_FLOAT(exp_avg_scale);
  CHECK_CUDA_TENSOR_FLOAT(exp_avg_sq_scale);
  CHECK_CUDA_TENSOR_FLOAT(exp_avg_qmap);
  CHECK_CUDA_TENSOR_FLOAT(exp_avg_sq_qmap);

  int64_t num_elem = p.numel();
  AT_ASSERTM(exp_avg.numel() == num_elem / 2,
             "number of elements in exp_avg and p tensors should be equal");
  AT_ASSERTM(exp_avg_sq.numel() == num_elem / 2,
             "number of elements in exp_avg_sq and p tensors should be equal");
  AT_ASSERTM(g.numel() == num_elem,
             "number of elements in g and p tensors should be equal");

  fused_adamw4bit_cuda(p, g, exp_avg, exp_avg_sq,
                       exp_avg_scale, exp_avg_sq_scale, exp_avg_qmap, exp_avg_sq_qmap,
                       beta1, beta2, lr, weight_decay, eps, step);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adamw_single_tensor", &adamw_single_tensor);
    m.def("adamw4bit_single_tensor", &adamw4bit_single_tensor);
}
