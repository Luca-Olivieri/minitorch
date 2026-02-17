#include "tensor_nodes.h"
#include "backward_ops.h"

#include "tensors.h"

TensorNode::TensorNode(
        const std::vector<size_t> shape,
        const float value,
        const bool requires_grad
):
    m_storage{ shape, value },
    m_grad_fn{ nullptr },
    m_grad{ nullptr },
    m_requires_grad{ requires_grad } {}

TensorNode::TensorNode(
        TensorStorage&& storage,
        const bool requires_grad
):
    m_storage{ std::move(storage) },
    m_grad_fn{ nullptr },
    m_grad{ nullptr },
    m_requires_grad{ requires_grad } {}
