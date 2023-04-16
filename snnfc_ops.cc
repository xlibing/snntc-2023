#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("SnnFc")
    .Attr("T: {int32, float}")
    .Input("in: T")
    .Input("weight: T")
    .Input("params: T")  // important parameters
    .Output("out: T")

    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));

    shape_inference::ShapeHandle weight_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_shape));

    shape_inference::DimensionHandle batch = c->Dim(input_shape, 0);
    shape_inference::DimensionHandle activations = c->Dim(weight_shape, 1);

    // output dim [2*B, J], where 2nd half is to save redundant params for grad calculation
    TF_RETURN_IF_ERROR(c->Add(batch, batch, &batch));
    c->set_output(0, c->Matrix(batch, activations));
//    c->set_output(0, c->MakeShape({c->UnknownDim(), activations}));

    return Status::OK();
    });
