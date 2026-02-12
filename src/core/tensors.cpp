#include "tensors.h"
#include "tensor_nodes.h"

Tensor::Tensor(
    std::vector<size_t> shape,
    float value
) {
    m_node = std::make_shared<TensorNode>(
        shape,
        value
    );
}

Tensor::Tensor(
    std::shared_ptr<TensorNode> node
): m_node{node} {}

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

void Tensor::fill_inplace(
    float value
) {
    m_node->fill_inplace(value);
}

void Tensor::linspace_inplace(
    float start,
    float end
) {
    m_node->linspace_inplace(start, end);
}

Tensor Tensor::linspace(
    float start,
    float end
) {
    return m_node->linspace(start, end);
}

Tensor Tensor::linspace(
    std::vector<size_t> shape,
    float start,
    float end
) {
    return TensorNode::linspace(std::move(shape), start, end);
}

Tensor Tensor::fill(
    float value
) {
    return m_node->fill(value);
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

Tensor Tensor::operator/(
    const Tensor& other
) {
    return (*m_node) / other;
}

Tensor Tensor::pow(
    const Tensor& other
) {
    return m_node->pow(other);
}

Tensor Tensor::log() {
    return m_node->log();
}

Tensor Tensor::sum(
    const size_t dim
) {
    return m_node->sum(*this, dim);
}

Tensor Tensor::grad() const {
    return Tensor(m_node->m_grad);
}

void Tensor::zero_grad() {
    m_node->zero_grad();
}

const std::vector<size_t>& Tensor::shape() {
    return m_node->m_storage.m_shape;
}

void Tensor::backward(bool create_graph) {
    m_node->backward(create_graph);
}

void Tensor::accumulate_grad(const Tensor& gradient, bool create_graph) {
    m_node->accumulate_grad(gradient, create_graph);
}
