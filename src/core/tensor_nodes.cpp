#include "tensor_nodes.h"
#include "backward_ops.h"

TensorNode::TensorNode(
    std::vector<size_t> shape
):
    m_storage{ shape },
    m_bw_op{ nullptr },
    m_grad{ nullptr } ,
    m_requires_grad { true } {}

TensorNode::TensorNode(
    TensorStorage& storage
):
    m_storage{ std::move(storage) },
    m_bw_op{ nullptr },
    m_grad{ nullptr } ,
    m_requires_grad { true } {}

std::ostream& operator<<(std::ostream& os, const TensorNode& tensor){
    return os << tensor.m_storage;
}

float& TensorNode::operator[](const std::vector<size_t>& md_index) {
    if (m_storage.m_shape.empty()) { // Scalar case
        throw std::invalid_argument(std::format("\nScalar tensor cannot be access by index. Got index {}", md_index));
    }
    return m_storage.get_entry_ref((md_index));
}

float& TensorNode::item() {
    return m_storage.item();
}

void TensorNode::fill(
    float value
) {
    m_storage.fill(value);
}

void TensorNode::linspace(
    float start,
    float end
) {
    m_storage.linspace(start, end);
}

bool TensorNode::is_contiguous() {
    return m_storage.is_contiguous();
}

Tensor TensorNode::operator+(
    const Tensor& other
) {
    return apply_op_ag<TensorStorage::s_add, BackwardAdd>(other);
}

void TensorNode::operator+=(
    const Tensor& other
) {
    TensorStorage::s_add_inplace(m_storage, other.m_node->m_storage);
}

Tensor TensorNode::operator-() {
    return apply_op_ag<TensorStorage::s_minus, BackwardMinus>();
}

Tensor TensorNode::operator-(
    const Tensor& other
) {
    return apply_op_ag<TensorStorage::s_sub, BackwardSub>(other);
}

Tensor TensorNode::operator*(
    const Tensor& other
) {
    return apply_op_ag<TensorStorage::s_mult, BackwardMult>(other);
}

Tensor TensorNode::pow(
    const Tensor& other
) {
    return apply_op_ag<TensorStorage::s_pow, BackwardPow>(other);
}

Tensor TensorNode::log() {
    return apply_op_ag<TensorStorage::s_log, BackwardLog>();
}

void TensorNode::reset_grad() {
    m_grad->fill(0);
    m_grad->m_bw_op = nullptr;
    m_grad->m_grad = nullptr;
}

void TensorNode::zero_grad() {
    if (m_grad) {
        reset_grad();
    }
    if (m_bw_op) {
        m_bw_op->reset_all_grads();
    }
}

void TensorNode::backward() {
    m_grad = std::make_shared<TensorNode>(m_storage.m_shape);
    m_grad->fill(1.0f);
    backprop();
}

void TensorNode::backprop() {
    if (m_bw_op) {
        m_bw_op->init_operands_grad_if_none();
        m_bw_op->compute_operands_grad(Tensor(shared_from_this()));
        m_bw_op->backprop();
    }
}
