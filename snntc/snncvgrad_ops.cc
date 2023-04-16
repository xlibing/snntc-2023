#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("SnnCvGrad")
    .Attr("T: {int32, float}")
    .Input("grad: T")
    .Input("in: T")
    .Input("weight: T")
    .Input("params: T")
    .Input("oldout: T")
    .Output("grad_input: T")
    .Output("grad_weight: T")
    .Output("grad_params: T")

    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));  // grad of inputs
      c->set_output(1, c->input(2));  // grad of weights
      c->set_output(2, c->input(3));  // grad of params, useless, et to 0

      return Status::OK();
    });
