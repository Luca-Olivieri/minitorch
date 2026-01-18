#include <iostream>
#include <format>
#include <string>
#include <iomanip>
#include <stdexcept>

#include "tensors.h"

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector){
    std::string out_string = "[";
    for (size_t i { 0 }; i < vector.size(); i++) {
        out_string += std::to_string(vector[i]);
        if (i != vector.size()-1) {
            out_string += ", ";
        }
    }
    out_string += "]";
    return os << out_string;
}

Tensor::Tensor(
    std::initializer_list<size_t> shape
): m_shape(shape) {
    size_t size { 1 };
    for (size_t dim : shape) {
        size *= dim; 
    }
    m_size = size;
    m_strides = Tensor::init_strides(m_shape);
    m_flat_data = std::vector<float>(size, 0.0f);
}

std::vector<size_t> Tensor::init_strides(
    const std::vector<size_t>& shape
) {
    std::vector<size_t> strides(shape.size(), 0.0f);
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
    // Error checking
    if (md_index.size() != m_shape.size()) {
        throw std::runtime_error(std::format("\nTensor of shape {} accessed at partial index {}", m_shape, md_index));
    }
    for (size_t i { 0 }; i < m_shape.size(); i++) {
        if (md_index[i] >= m_shape[i]) {
            throw std::out_of_range(std::format("\nTensor of shape {} accessed out-of-bounds at index {}", m_shape, md_index));
        }
        if (md_index[i] < 0) {
            // TODO: right now, indices cannot be negative because size_t rolls up.
            throw std::out_of_range(std::format("\nTensor accessed at negative index {}", md_index));
        }
    }
    // Flat index computation
    size_t flat_index { 0 };
    for (size_t i { 0 }; i < m_shape.size(); i++) {
        flat_index += m_strides[i]*md_index[i];
    }
    return flat_index;
}

void Tensor::fill(
    float value
) {
    for (size_t i = 0; i < m_size; i++) {
        m_flat_data[i] = value;
    }
}

// Helper functoin for the Tensor cout print
void print_recursive(
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
    os << std::format("Tensor(shape={}, dtype=float,\n       ", tensor.m_shape);
    if (tensor.m_shape.empty()) {
        if (!tensor.m_flat_data.empty())
             os << std::fixed << std::setprecision(4) << tensor.m_flat_data[0];
    } else {
        print_recursive(os, tensor, 0, 0, 7);
    }
    os << ")";
    return os;
}
