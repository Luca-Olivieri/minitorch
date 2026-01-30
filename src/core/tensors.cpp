#include <iostream>
#include <format>
#include <string>
#include <iomanip>
#include <stdexcept>

#include "tensors.h"

Tensor::Tensor(
    std::vector<size_t> shape
): m_shape(shape) {
    // shape validation
    
    for (size_t dim : shape) {
        if (dim == 0) {
            throw std::invalid_argument(std::format("\nTensor shape must have positive dimensions. Got {}.", m_shape));
        }
    }
    
    size_t numel { 1 };
    if (m_shape.empty()) {
        m_numel = 1;
    }
    else {
        for (size_t dim : shape) {
            numel *= dim;
        }
        m_numel = numel;
    }
    m_strides = Tensor::init_strides(m_shape);
    m_flat_data = std::vector<float>(numel, 0.0f);
}

std::vector<size_t> Tensor::init_strides(
    const std::vector<size_t>& shape
) {
    std::vector<size_t> strides(shape.size(), 0);
    size_t curr_stride {1};
    // very clunky loop signature to have an unsigned int check for >= 0
    for (size_t i { shape.size() }; i-- > 0; ) {
        strides[i] = curr_stride;
        curr_stride *= shape[i];
    }
    return strides;
}

size_t Tensor::get_flat_index(
    const std::vector<size_t>& md_index
) {
    if (md_index.size() != m_shape.size()) {
        throw std::invalid_argument(std::format("Index size {} does not match tensor shape size {}.", md_index.size(), m_shape.size()));
    }
    // Flat index computation
    size_t flat_index { 0 };
    for (size_t i { 0 }; i < m_shape.size(); i++) {
        if (md_index[i] >= m_shape[i]) {
            throw std::out_of_range(std::format("Index {} out of bounds for dimension {} of size {}.", md_index[i], i, m_shape[i]));
        }
        flat_index += m_strides[i]*md_index[i];
    }
    if (flat_index >= m_numel) {
        throw std::out_of_range(std::format("Tensor flat data of size{} accessed at depth {}, of index {}.", m_numel, flat_index, md_index));
    }
    return flat_index;
}

float& Tensor::operator[](const std::vector<size_t>& md_index) {
    // Scalar case
    if (m_shape.empty()) {
        throw std::invalid_argument(std::format("\nScalar tensor cannot be access by index. Got index {}", md_index));
    }
    return m_flat_data[get_flat_index(md_index)];
}

void Tensor::fill(
    float value
) {
    for (size_t i = 0; i < m_numel; i++) {
        m_flat_data[i] = value;
    }
}

void Tensor::linspace(
    float start,
    float end
) {
    float delta = (end-start)/(static_cast<float>(m_numel-1));
    for (size_t i = 0; i < m_numel; i++) {
        m_flat_data[i] = start + static_cast<float>(i)*delta;
    }
}

float Tensor::item() {
    if (m_numel != 1) {
        throw std::runtime_error(std::format("Cannot call item() on a non-singleton tensor (shape {}).", m_shape));
    }
    return m_flat_data[0];
}

Tensor Tensor::get_slice(
    size_t slice_index
) {
    if (m_shape.empty()) {
        throw std::runtime_error("Cannot slice a scalar tensor.");
    }
    if (slice_index >= m_shape[0]) {
        throw std::out_of_range(std::format("Slice index {} out of bounds for dimension 0 of size {}.", slice_index, m_shape[0]));
    }
    
    std::vector<size_t> shape_without_first(m_shape.begin() + 1, m_shape.end());
    Tensor sliced_tensor(shape_without_first);
    
    size_t start_offset = slice_index * m_strides[0];
    for (size_t i = 0; i < m_strides[0]; i++) {
        sliced_tensor.m_flat_data[i] = m_flat_data[start_offset + i];
    }
    return sliced_tensor;
}

void Tensor::reshape(
    std::vector<size_t> shape
) {
    for (size_t dim : shape) {
        if (dim == 0) {
            throw std::invalid_argument(std::format("\nTensor shape must have positive dimensions. Got {}.", shape));
        }
    }

    size_t new_numel { 1 };
    if (!shape.empty()) {
        for (size_t dim : shape) {
            new_numel *= dim;
        }
    }

    if (new_numel != m_numel) {
        throw std::invalid_argument(std::format("Cannot reshape tensor of size {} into shape {}.", m_numel, shape));
    }
    
    m_shape = shape;
    m_strides = Tensor::init_strides(m_shape);
}

void Tensor::transpose(
    size_t dim_1,
    size_t dim_2
) {
    size_t temp_stride = m_strides[dim_1];
    m_strides[dim_1] = m_strides[dim_2];
    m_strides[dim_2] = temp_stride;

    size_t temp_dim = m_shape[dim_1];
    m_shape[dim_1] = m_shape[dim_2];
    m_shape[dim_2] = temp_dim;
}

// Helper functoin for the Tensor cout print
static void print_recursive(
    std::ostream& os,
    const Tensor& tensor,
    size_t dim_index,
    size_t offset,
    int indent
) {
    size_t dim_size = tensor.m_shape[dim_index];
    size_t stride = tensor.m_strides[dim_index];
    
    if (dim_index == tensor.m_shape.size() - 1) {
        os << "[";
        for (size_t i = 0; i < dim_size; ++i) {
            float val = tensor.m_flat_data[offset + i * stride];
            os << std::fixed << std::setprecision(4) << val;
            if (i < dim_size - 1) {
                os << ", ";
            }
        }
        os << "]";
    } else {
        os << "[";
        for (size_t i = 0; i < dim_size; ++i) {
            if (i > 0) {
                os << ",";
                size_t newlines = tensor.m_shape.size() - dim_index - 1;
                for (size_t nl = 0; nl < newlines; ++nl) os << "\n";
                for (int k = 0; k < indent + 1; ++k) os << " ";
            }
            print_recursive(
                os,
                tensor,
                dim_index + 1,
                offset + i * stride,
                indent + 1
            );
        }
        os << "]";
    }
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor){
    os << std::format("Tensor(shape={}, dtype=float,\n       data=", tensor.m_shape);
    if (tensor.m_shape.empty()) {
        if (!tensor.m_flat_data.empty())
             os << std::fixed << std::setprecision(4) << tensor.m_flat_data[0];
    } else {
        print_recursive(os, tensor, 0, 0, 12);
    }
    os << ")";
    return os;
}
