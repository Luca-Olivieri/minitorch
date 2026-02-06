#include "tensors.h"

Tensor::Tensor(
    std::vector<size_t> shape
) {
    node = std::make_shared<TensorNode>(shape);
}

Tensor::Tensor(): node{nullptr} {}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor){
    return os << *tensor.node;
}

float& Tensor::operator[](
    const std::vector<size_t>& md_index
) {
    return (*node)[md_index];
}

float& Tensor::item() {
    return node->item();
}

void Tensor::fill(
    float value
) {
    node->fill(value);
}

void Tensor::linspace(
    float start,
    float end
) {
    node->linspace(start, end);
}

bool Tensor::is_contiguous() {
    return node->is_contiguous();
}

Tensor Tensor::operator+(
    Tensor& other
) {
    Tensor out {};
    out.node = std::make_shared<TensorNode>((*node) + (*other.node));
    return out;
}
