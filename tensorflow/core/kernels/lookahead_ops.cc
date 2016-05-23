

#include "tensorflow/core/kernels/lookahead_ops.h"

using namespace tensorflow;

template<typename T>
class LookaheadOp<T, 0> : public OpKernel {
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

    for (int batch = 0; batch < input_tensor.dim_size(0); batch++) {
      for (int t = 0; t < input_tensor.dim_size(1); t++) {
        for (int f = 0; f < input_tensor.dim_size(2); f++) {
          output(batch, t, f) = 0;
          for(int tau = 0; tau < filter_tensor.dim_size(0); tau++) {
            if(t + tau < input_tensor.dim_size(1)) output(batch, t, f) += input(batch, t + tau, f) * filter(tau, f);
          }
        }
      }
    }
  }

 private:
  int preserve_index_;
};

REGISTER_KERNEL_BUILDER(Name("Lookaheadcpu").Device(DEVICE_CPU), LookaheadOp<float, 0>);
