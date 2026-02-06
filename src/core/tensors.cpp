#include "tensors.h"
#include "backward_ops.h"

Tensor::Tensor(
    std::vector<size_t> shape
):
    m_value{ shape },
    m_bw_op{ nullptr },
    m_grad{ nullptr } ,
    m_requires_grad { true } {}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor){
    return os << tensor.m_value;
}

float& Tensor::operator[](const std::vector<size_t>& md_index) {
    if (m_value.m_shape.empty()) { // Scalar case
        throw std::invalid_argument(std::format("\nScalar tensor cannot be access by index. Got index {}", md_index));
    }
    return m_value.get_entry_ref((md_index));
}

float& Tensor::item() {
    return m_value.item();
}

void Tensor::fill(
    float value
) {
    m_value.fill(value);
}

void Tensor::linspace(
    float start,
    float end
) {
    m_value.linspace(start, end);
}

bool Tensor::is_contiguous() {
    return m_value.is_contiguous();
}

Tensor Tensor::operator+(
    const Tensor& other
) const {
    return apply_op_ag<TensorImpl::s_add, BackwardAdd>(other);
}

void Tensor::operator+=(
    const Tensor& other
) {
    TensorImpl::s_add_inplace(m_value, other.m_value);
}

Tensor Tensor::operator-() const {
    return apply_op_ag<TensorImpl::s_minus, BackwardMinus>();
}

Tensor Tensor::operator*(
    const Tensor& other
) const {
    return apply_op_ag<TensorImpl::s_mult, BackwardMult>(other);
}

Tensor Tensor::pow(
    const Tensor& other
) const {
    return apply_op_ag<TensorImpl::s_pow, BackwardPow>(other);
}

void Tensor::reset_grad() {
    m_grad->fill(0);
    m_grad->m_bw_op = nullptr;
    m_grad->m_grad = nullptr;
}

void Tensor::zero_grad() {
    if (m_grad) {
        reset_grad();
    }
    if (m_bw_op) {
        m_bw_op->reset_all_grads();
    }
}

void Tensor::backward() {
    m_grad = std::make_shared<Tensor>(m_value.m_shape);
    m_grad->fill(1.0f);
    backprop();
}

void Tensor::backprop() {
    if (m_bw_op) {
        m_bw_op->init_operands_grad_if_none();
        m_bw_op->compute_operands_grad(*this);
        m_bw_op->backprop();
    }
}
