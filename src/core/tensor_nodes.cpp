#include "tensor_nodes.h"
#include "backward_ops.h"

TensorNode::TensorNode(
    std::vector<size_t> shape
):
    m_value{ shape },
    m_bw_op{ nullptr },
    m_grad{ nullptr } ,
    m_requires_grad { true } {}

std::ostream& operator<<(std::ostream& os, const TensorNode& tensor){
    return os << tensor.m_value;
}

float& TensorNode::operator[](const std::vector<size_t>& md_index) {
    if (m_value.m_shape.empty()) { // Scalar case
        throw std::invalid_argument(std::format("\nScalar tensor cannot be access by index. Got index {}", md_index));
    }
    return m_value.get_entry_ref((md_index));
}

float& TensorNode::item() {
    return m_value.item();
}

void TensorNode::fill(
    float value
) {
    m_value.fill(value);
}

void TensorNode::linspace(
    float start,
    float end
) {
    m_value.linspace(start, end);
}

bool TensorNode::is_contiguous() {
    return m_value.is_contiguous();
}

std::shared_ptr<TensorNode> TensorNode::operator+(
    TensorNode& other
) {
    return apply_op_ag<TensorStorage::s_add, BackwardAdd>(other);
}

void TensorNode::operator+=(
    TensorNode& other
) {
    TensorStorage::s_add_inplace(m_value, other.m_value);
}

std::shared_ptr<TensorNode> TensorNode::operator-() {
    return apply_op_ag<TensorStorage::s_minus, BackwardMinus>();
}

std::shared_ptr<TensorNode> TensorNode::operator-(
    TensorNode& other
) {
    return apply_op_ag<TensorStorage::s_sub, BackwardSub>(other);
}

std::shared_ptr<TensorNode> TensorNode::operator*(
    TensorNode& other
) {
    return apply_op_ag<TensorStorage::s_mult, BackwardMult>(other);
}

std::shared_ptr<TensorNode> TensorNode::pow(
    TensorNode& other
) {
    return apply_op_ag<TensorStorage::s_pow, BackwardPow>(other);
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
    m_grad = std::make_shared<TensorNode>(m_value.m_shape);
    m_grad->fill(1.0f);
    backprop();
}

void TensorNode::backprop() {
    if (m_bw_op) {
        m_bw_op->init_operands_grad_if_none();
        m_bw_op->compute_operands_grad(*this);
        m_bw_op->backprop();
    }
}
