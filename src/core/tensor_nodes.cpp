#include "tensor_nodes.h"
#include "backward_ops.h"

#include "tensors.h"

TensorNode::TensorNode(
    std::vector<size_t> shape,
    float value
):
    m_storage{ shape, value },
    m_bw_op{ nullptr },
    m_grad{ nullptr } {}

TensorNode::TensorNode(
    TensorStorage storage
):
    m_storage{ std::move(storage) },
    m_bw_op{ nullptr },
    m_grad{ nullptr } {}
