#include "tensorflow/core/kernels/lookahead_ops.h"

using namespace tensorflow;

template<typename T>
__global__ void kernel(int dim_t, int dim_f, int dim_tau, const T* input, const T* filter, T* output) {
  int t = blockIdx.x;
  int f = threadIdx.x;
  output[t * dim_f + f] = 0;
  for(int tau = 0; tau < dim_tau; tau++) {
    if(t + tau < dim_t) output[t* dim_f + f] += input[(t + tau) * dim_f + f] * filter[tau * dim_f + f];
  }
}

template<typename T>
class LookaheadOp<T, 1> : public OpKernel {
 public:
  explicit LookaheadOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt}, {dt}));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.tensor<T, 3>();
    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.matrix<T>();

    // Create output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template tensor<T, 3>();
    int batch_size = input_tensor.dim_size(0);
    cudaStream_t stream[batch_size];
    int dim_t = input_tensor.dim_size(1);
    int dim_f = input_tensor.dim_size(2);
    int dim_tau = filter_tensor.dim_size(0);
    for(int i = 0; i < batch_size; i++) {
      cudaStreamCreate(&stream[i]);
    }
    for(int i = 0; i < batch_size; i++) {
      kernel<T><<<dim_t, dim_f, 0, stream[i]>>>(dim_t, dim_f, dim_tau, &input(i, 0, 0), &filter(0, 0), &output(i, 0, 0));
    }
    for(int i = 0; i < batch_size; i++) {
      cudaStreamSynchronize(stream[i]);
      cudaStreamDestroy(stream[i]);
    }
  }

 private:
  int preserve_index_;
};

REGISTER_KERNEL_BUILDER(Name("Lookaheadgpu").Device(DEVICE_GPU), LookaheadOp<float, 1>);
