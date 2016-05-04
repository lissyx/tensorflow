

#include "tensorflow/core/kernels/lookahead_grad_op.h"

using namespace tensorflow;

template<typename T>
class LookaheadGradInputOp<T, 0> : public OpKernel {
 public:
  explicit LookaheadGradInputOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt, dt}, {dt}));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    printf("grad input cpu\n");
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.tensor<T, 3>();

    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.tensor<T, 3>();

    const Tensor& backprop_output_tensor = context->input(2);
    auto backprop_output = backprop_output_tensor.tensor<T, 3>();

    // Check that preserve_index is in range

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template tensor<T, 3>();

    for (int batch = 0; batch < input_tensor.dim_size(0); batch++) {
      for (int input_x = 0; input_x < input_tensor.dim_size(1); input_x++) {
        for (int input_y = 0; input_y < input_tensor.dim_size(2); input_y++) {
          output(batch, input_x, input_y) = 0;
          for (int input_begin = 0; input_begin < filter_tensor.dim_size(1); input_begin++) {
            int index = input_begin + input_y - filter_tensor.dim_size(1) + 1;
            if (index >= 0) {
              output(batch, input_x, input_y) += backprop_output(batch, input_x, index) / filter(batch, filter_tensor.dim_size(1) - 1 - input_begin, input_x);
            }
          }
        }
      }
    }
  }

 private:
  int preserve_index_;
};

template<typename T>
class LookaheadGradFilterOp<T, 0> : public OpKernel {
 public:
  explicit LookaheadGradFilterOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt, dt}, {dt}));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    printf("grad filter cpu\n");
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.tensor<T, 3>();

    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.tensor<T, 3>();

    const Tensor& backprop_output_tensor = context->input(2);
    auto backprop_output = backprop_output_tensor.tensor<T, 3>();

    // Check that preserve_index is in range

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, filter_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template tensor<T, 3>();

    for (int batch = 0; batch < filter_tensor.dim_size(0); batch++) {
      for (int input_y = 0; input_y < filter_tensor.dim_size(2); input_y++) {
        for (int input_x = 0; input_x < filter_tensor.dim_size(1); input_x++) {
          output(batch, input_x, input_y) = 0;
          for (int input_begin = 0; input_begin < input_tensor.dim_size(2) - input_x; input_begin++) {
            output(batch, input_x, input_y) += backprop_output(batch, input_y, input_begin) / input(batch, input_y, input_x + input_begin);
          }
        }
      }
    }
  }

 private:
  int preserve_index_;
};

REGISTER_KERNEL_BUILDER(Name("Lookaheadgradinputcpu").Device(DEVICE_CPU), LookaheadGradInputOp<float, 0>);
REGISTER_KERNEL_BUILDER(Name("Lookaheadgradfiltercpu").Device(DEVICE_CPU), LookaheadGradFilterOp<float, 0>);
