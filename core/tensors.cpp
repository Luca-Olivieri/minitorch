#include <iostream>
#include <format>
#include <string>
#include <iomanip>

#include "tensors.h"

std::ostream& operator<<(std::ostream& os, const std::vector<float>& vector){
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
    m_strides = Tensor::init_strides(m_shape);
    m_flat_data = Tensor::init_flat_data(m_shape);
}

std::vector<float> Tensor::init_flat_data(
        const std::vector<size_t>& shape
) {
    size_t total_len { 1 };
    for (size_t dim : shape) {
        total_len *= dim; 
    }
    return std::vector<float>(total_len, 0.0f);
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
    // TODO add the bounds check
    size_t flat_index { 0 };
    for (size_t i { 0 }; i < md_index.size(); i++) {
        flat_index += m_strides[i]*md_index[i];
    }
    return flat_index;
}

// for the Tensor cout print
void print_recursive(std::ostream& os, const Tensor& tensor, size_t dim_index, size_t offset, int indent) {
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
            print_recursive(os, tensor, dim_index + 1, offset + i * stride, indent + 1);
        }
        os << "]";
    }
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor){
    os << "tensor(";
    if (tensor.m_shape.empty()) {
        if (!tensor.m_flat_data.empty())
             os << std::fixed << std::setprecision(4) << tensor.m_flat_data[0];
    } else {
        print_recursive(os, tensor, 0, 0, 7);
    }
    os << ")";
    return os;
}
