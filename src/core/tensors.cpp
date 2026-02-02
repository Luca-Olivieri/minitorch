#include "tensors.h"

Tensor::Tensor(
    std::vector<size_t> shape
): m_value(shape) {}

float& Tensor::operator[](const std::vector<size_t>& md_index) {
    // Scalar case
    if (m_value.m_shape.empty()) {
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

std::ostream& operator<<(std::ostream& os, const Tensor& tensor){
    return os << tensor.m_value;
}

