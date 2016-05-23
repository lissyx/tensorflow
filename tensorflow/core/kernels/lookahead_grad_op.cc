

#include "tensorflow/core/kernels/lookahead_grad_op.h"

using namespace tensorflow;

template<typename T>
class LookaheadGradOp<T, 0> : public OpKernel {
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

    for (int batch = 0; batch < input_tensor.dim_size(0); batch++) {
      for (int t = 0; t < input_tensor.dim_size(1); t++) {
        for (int f = 0; f < input_tensor.dim_size(2); f++) {
          output(batch, t, f) = 0;
          for (int input_begin = 0; input_begin < filter_tensor.dim_size(0); input_begin++) {
            int index = input_begin + t - filter_tensor.dim_size(0) + 1;
            int filter_idx = filter_tensor.dim_size(0) - 1 - input_begin;
            if (index >= 0 && filter_idx >= 0) {
              output(batch, t, f) += backprop_output(batch, index, f) * filter(filter_idx, f);
            }
          }
        }
      }
    }
    // Create filter grad output tensor
    OP_REQUIRES_OK(context, context->allocate_output(1, filter_tensor.shape(),
                                                     &output_tensor));
    auto output2 = output_tensor->template matrix<T>();

    for (int f = 0; f < filter_tensor.dim_size(1); f++) {
      for (int t = 0; t < filter_tensor.dim_size(0); t++) {
        output2(t, f) = 0;
      }
    }
    for (int batch = 0; batch < input_tensor.dim_size(0); batch++) {
      for (int f = 0; f < filter_tensor.dim_size(1); f++) {
        for (int tau = 0; tau < filter_tensor.dim_size(0); tau++) {
          for (int t = 0; t < input_tensor.dim_size(1) - tau; t++) {
            output2(tau, f) += backprop_output(batch, t, f) * input(batch, t + tau, f);
          }
        }
      }
    }
  }

 private:
  int preserve_index_;
};

REGISTER_KERNEL_BUILDER(Name("Lookaheadgradcpu").Device(DEVICE_CPU), LookaheadGradOp<float, 0>);
