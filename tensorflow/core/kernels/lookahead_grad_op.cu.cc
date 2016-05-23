

#include "tensorflow/core/kernels/lookahead_grad_op.h"

using namespace tensorflow;

template<typename T>
__global__ void kernel_grad_input(int dim_t, int dim_f, int dim_tau, const T* filter, const T* back_prop, T* output) {
  int f = threadIdx.x;
  int t = blockIdx.x;
  output[t * dim_f + f] = 0;
  for(int input_begin = 0; input_begin < dim_tau; input_begin++) {
    int index = input_begin + t - dim_tau + 1;
    int filter_idx = dim_tau - 1 - input_begin;
    if (index >= 0 && filter_idx >= 0) {
      output[t * dim_f + f] += back_prop[index * dim_f + f] * filter[filter_idx * dim_f + f];
    }
  }
}

template<typename T>
__global__ void kernel_grad_filter(int dim_tau, int dim_f, int input_size, const T* input, const T* back_prop, T* output) {
  int f = threadIdx.x;
  int tau = blockIdx.x;
  for (int t = 0; t < input_size - tau; t++) {
    output[tau * dim_f + f] += back_prop[t * dim_f + f] * input[(t + tau) * dim_f + f];
  }
}

template<typename T>
class LookaheadGradOp<T, 1> : public OpKernel {
 public:
  explicit LookaheadGradOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt, dt}, {dt, dt}));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.tensor<T, 3>();

    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.matrix<T>();

    const Tensor& backprop_output_tensor = context->input(2);
    auto backprop_output = backprop_output_tensor.tensor<T, 3>();

    // Create input grad output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template tensor<T, 3>();

    int dim_batch = input_tensor.dim_size(0);
    int dim_t = input_tensor.dim_size(1);
    int dim_f = input_tensor.dim_size(2);
    int dim_tau = filter_tensor.dim_size(0);
    cudaStream_t stream[dim_batch];
    for (int i = 0; i < dim_batch; i++) {
      cudaStreamCreate(&stream[i]);
    }
    for (int i = 0; i < dim_batch; i++) {
      kernel_grad_input<T><<<dim_t, dim_f, 0, stream[i]>>>(dim_t, dim_f, dim_tau, &filter(0, 0), &backprop_output(i, 0, 0), &output(i, 0, 0));
    }
    for (int i = 0; i < dim_batch; i++) {
      cudaStreamSynchronize(stream[i]);
      cudaStreamDestroy(stream[i]);
    }
    // Create filter grad output tensor
    OP_REQUIRES_OK(context, context->allocate_output(1, filter_tensor.shape(),
                                                     &output_tensor));
    auto output2 = output_tensor->template matrix<T>();

    cudaStream_t streamx;
    cudaStreamCreate(&streamx);
    cudaMemset(&output2(0, 0), 0, dim_tau * dim_f * sizeof(T));
    for (int i = 0; i < dim_batch; i++) {
      kernel_grad_filter<T><<<dim_tau, dim_f, 0, streamx>>>(dim_tau, dim_f, dim_t, &input(i, 0, 0), &backprop_output(i, 0, 0), &output2(0, 0));
      cudaStreamSynchronize(streamx);
    }
    cudaStreamDestroy(streamx);
  }

 private:
  int preserve_index_;
};

REGISTER_KERNEL_BUILDER(Name("Lookaheadgradgpu").Device(DEVICE_GPU), LookaheadGradOp<float, 1>);
