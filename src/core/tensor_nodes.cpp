#include "tensor_nodes.h"
#include "backward_ops.h"

#include "tensors.h"

TensorNode::TensorNode(
        const std::vector<size_t> shape,
        const float value
):
    m_storage{ shape, value },
    m_grad_fn{ nullptr },
    m_grad{ nullptr } {}

TensorNode::TensorNode(
        TensorStorage&& storage
):
    m_storage{ std::move(storage) },
    m_grad_fn{ nullptr },
    m_grad{ nullptr } {}
