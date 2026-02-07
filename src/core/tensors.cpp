#include "tensors.h"
#include "tensor_nodes.h"

Tensor::Tensor(
    std::vector<size_t> shape
) {
    m_node = std::make_shared<TensorNode>(shape);
}

Tensor::Tensor(
    std::shared_ptr<TensorNode> node
): m_node{node} {}

Tensor::Tensor(): m_node{nullptr} {}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor){
    return os << *tensor.m_node;
}

float& Tensor::operator[](
    const std::vector<size_t>& md_index
) {
    return (*m_node)[md_index];
}

float& Tensor::item() {
    return m_node->item();
}

void Tensor::fill(
    float value
) {
    m_node->fill(value);
}

void Tensor::linspace(
    float start,
    float end
) {
    m_node->linspace(start, end);
}

bool Tensor::is_contiguous() {
    return m_node->is_contiguous();
}

Tensor Tensor::operator+(
    const Tensor& other
) {
    return (*m_node) + other;
}

void Tensor::operator+=(
    const Tensor& other
) {
    *m_node += other;
}

Tensor Tensor::operator-(
) {
    return -(*m_node);
}

Tensor Tensor::operator-(
    const Tensor& other
) {
    return (*m_node) - other;
}

Tensor Tensor::operator*(
    const Tensor& other
) {
    return (*m_node) * other;
}

Tensor Tensor::pow(
    const Tensor& other
) {
    return m_node->pow(other);
}

Tensor Tensor::grad() {
    return Tensor(m_node->m_grad);
}

void Tensor::zero_grad() {
    m_node->zero_grad();
}

const std::vector<size_t>& Tensor::shape() {
    return m_node->m_value.m_shape;
}

void Tensor::backward() {
    m_node->backward();
}

void Tensor::backprop() {
    m_node->backprop();
}
