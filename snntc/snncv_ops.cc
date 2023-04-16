#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("SnnCv")
    .Attr("T: {int32, float}")
    .Input("in: T")
    .Input("weight: T")
    .Input("params: T")  // important parameters
    .Output("out: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    shape_inference::ShapeHandle input_shape;  // input has rank 4 [B, H, W, C]
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

    shape_inference::ShapeHandle weight_shape; // weight has rank 2 [Kh*Kw*C+1, Kc]
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_shape));

    shape_inference::DimensionHandle batch = c->Dim(input_shape, 0);
    shape_inference::DimensionHandle activations = c->Dim(weight_shape, 1);

    // for tensorflow to obtain shape information
    // output dim should be [B, Nh, Nw, Kc],
    // but we make it [2*B, Nh, Nw, Kc] where the 2nd half is to save redundant params for grad calculation
    TF_RETURN_IF_ERROR(c->Add(batch, batch, &batch));
    c->set_output(0, c->MakeShape({batch, c->UnknownDim(), c->UnknownDim(), activations}));

    return Status::OK();
    });

// sample shape inference commands for future reference
// c->set_output(0, c->Matrix(batch, activations));
//  c->set_output(0, c->input(0));
//shape_inference::DimensionHandle batch3;  // Three times batch as output batch
//c->Multiply(batch, 3, &batch3);