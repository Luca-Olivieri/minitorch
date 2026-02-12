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

TensorNode::~TensorNode() = default;

std::ostream& operator<<(std::ostream& os, const TensorNode& tensor){
    return os << tensor.m_storage;
}

float& TensorNode::operator[](const std::vector<size_t>& md_index) {
    if (m_storage.m_shape.empty()) { // Scalar case
        throw std::invalid_argument(std::format("\nScalar tensor cannot be access by index. Got index {}", md_index));
    }
    return m_storage.get_entry_ref((md_index));
}
