#include <iostream>
#include <format>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <cmath>

#include "tensors.h"

Tensor::Tensor(
    std::vector<size_t> shape
): m_shape(shape), m_offset(0), m_grad{nullptr}, m_grad_fn{nullptr} {
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
    m_flat_data = std::vector<float>(m_numel, 0.0f);
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

size_t Tensor::get_flat_index_from_md(
    const std::vector<size_t>& md_index
) const {
    if (md_index.size() != m_shape.size()) {
        throw std::invalid_argument(std::format("Index size {} does not match tensor shape size {}.", md_index.size(), m_shape.size()));
    }
    // Flat index computation
    size_t flat_index { m_offset };
    for (size_t i { 0 }; i < m_shape.size(); i++) {
        if (md_index[i] >= m_shape[i]) {
            throw std::out_of_range(std::format("Index {} out of bounds for dimension {} of size {}.", md_index[i], i, m_shape[i]));
        }
        flat_index += m_strides[i]*md_index[i];
    }
    return flat_index;
}

size_t Tensor::get_flat_index_from_logical(
    size_t l_index
) const {
    if (l_index >= m_numel) {
        throw std::out_of_range(std::format("Index {} out of bounds for tensor of size {}.", l_index, m_numel));
    }
    
    size_t offset = m_offset;
    size_t current_index = l_index;

    for (size_t i = m_shape.size(); i-- > 0; ) {
        size_t dim_size = m_shape[i];
        size_t coord = current_index % dim_size;
        current_index /= dim_size;
        offset += coord * m_strides[i];
    }
    return offset;
}

float& Tensor::get_entry_ref(
    size_t l_index
) {
    return m_flat_data[get_flat_index_from_logical(l_index)];
}

float& Tensor::operator[](const std::vector<size_t>& md_index) {
    // Scalar case
    if (m_shape.empty()) {
        throw std::invalid_argument(std::format("\nScalar tensor cannot be access by index. Got index {}", md_index));
    }
    return m_flat_data[get_flat_index_from_md(md_index)];
}

bool Tensor::is_contiguous() {
    std::vector<size_t> contiguous_strides = Tensor::init_strides(m_shape);
    for (size_t i = 0; i < m_shape.size(); ++i) {
        if (m_shape[i] == 1) continue; // singleton dimensions do not affect contiguity
        if (m_strides[i] != contiguous_strides[i]) return false;
    }
    return true;
}

void Tensor::fill(
    float value
) {
    for (size_t i = 0; i < m_numel; i++) {
        get_entry_ref(i) = value;
    }
}

void Tensor::linspace(
    float start,
    float end
) {
    float delta = (end-start)/(static_cast<float>(m_numel-1));
    for (size_t i = 0; i < m_numel; i++) {
        get_entry_ref(i) = start + static_cast<float>(i)*delta;
    }
}

float& Tensor::item() {
    if (m_numel != 1) {
        throw std::runtime_error(std::format("Cannot call item() on a non-singleton tensor (shape {}).", m_shape));
    }
    return m_flat_data[m_offset];
}

void Tensor::slice(
    size_t dim,
    size_t slice_index
) {
    if (m_shape.empty()) {
        throw std::runtime_error("Cannot slice a scalar tensor.");
    }
    if (dim >= m_shape.size()) {
        throw std::out_of_range(std::format("Tensor has {} dimensions. Got slicing of dimension {}", m_shape.size(), dim));
    }
    if (slice_index >= m_shape[dim]) {
        throw std::out_of_range(std::format("Slice index {} out of bounds for dimension {} of size {}.", slice_index, dim, m_shape[dim]));
    }

    m_offset += slice_index*m_strides[dim];
    m_numel /= m_shape[dim];

    std::vector<size_t> new_shape(m_shape.size()-1); 
    std::vector<size_t> new_strides(m_strides.size()-1); 
    size_t j { 0 };
    for (size_t i = 0; i < m_shape.size(); i++) {
        if (i != dim) {
            new_shape[j] = m_shape[i];
            new_strides[j] = m_strides[i];
            j++;
        }
    }
    m_shape = new_shape;
    m_strides = new_strides;
}

void Tensor::dice(
    size_t dim,
    size_t index_start,
    size_t index_end
) {
    if (m_shape.empty()) {
        throw std::runtime_error("Cannot dice a scalar tensor.");
    }
    if (dim >= m_shape.size()) {
        throw std::out_of_range(std::format("Tensor has {} dimensions. Got dicing of dimension {}", m_shape.size(), dim));
    }
    if (index_start >= m_shape[dim] || index_end > m_shape[dim]) {
        throw std::out_of_range(std::format("Dice [{}, {}] out of bounds for dimension {} of size {}.", index_start, index_end, dim, m_shape[dim]));
    }
    if (index_start >= index_end) {
        throw std::out_of_range(std::format("Dice start ({}) should be less than to end ({}).", index_start, index_end));
    }
    
    m_offset += index_start*m_strides[dim];
    m_numel /= m_shape[dim];
    m_numel *= index_end-index_start;

    m_shape[dim] = index_end-index_start;
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

bool Tensor::are_shapes_equal(
    Tensor& a,
    Tensor& b
) {
    return a.m_shape == b.m_shape;
}

// Tensor Tensor::mult(
//     Tensor& a,
//     Tensor& b
// ) {
//     if (!are_shapes_equal(a, b)) {
//         throw std::invalid_argument(std::format("The operands of a multiplication should have the same shape. Got {} and {}", a.m_shape, b.m_shape));
//     }
//     Tensor new_tensor(a.m_shape);
//     for (size_t i = 0; i < a.m_numel; i++) {
//         new_tensor.m_flat_data[i] = a.get_entry_ref(i) * b.get_entry_ref(i);
//     }
//     new_tensor.m_grad_fn = std::make_shared<BackwardMult>(std::array<Tensor*, 2>({&a, &b}));
//     return new_tensor;
// }

Tensor Tensor::mult(
    Tensor& a,
    Tensor& b
) {
    constexpr size_t N = BackwardMult::N;
    return apply_op<N, BackwardMult>(
        std::array<Tensor*, N>{&a, &b},
        [](std::array<float, N> entries) { 
                return entries[0] * entries[1]; 
        }
    );
}

Tensor Tensor::add(
    Tensor& a,
    Tensor& b
) {
    constexpr size_t N = BackwardAdd::N;
    return apply_op<N, BackwardAdd>(
        std::array<Tensor*, N>{&a, &b},
        [](std::array<float, N> entries) { 
                return entries[0] + entries[1]; 
        }
    );
}

Tensor& Tensor::add_inplace(
    Tensor& a,
    Tensor& b
) {
    constexpr size_t N = BackwardAdd::N;
    return apply_op_inplace<N>(
        std::array<Tensor*, N>{&a, &b},
        [](std::array<float, N> entries) { 
                return entries[0] + entries[1]; 
        }
    );
}

Tensor Tensor::operator*(
    Tensor& other
) {
    return Tensor::mult(*this, other);
}

Tensor Tensor::operator+(
    Tensor& other
) {
    return Tensor::add(*this, other);
}

Tensor& Tensor::operator+=(
    Tensor& other
) {
    return Tensor::add_inplace(*this, other);
}

void Tensor::reset_grads() {
    if (m_grad)
        m_grad->fill(0.0f);
}

void Tensor::backward() {
    if (!m_grad)
        m_grad = std::make_shared<Tensor>(m_shape);
    m_grad->fill(1.0f);
    backprop();
}

void Tensor::backprop() {
    if (m_grad_fn) {
        m_grad_fn->backprop(*this);
    }
}

// Helper function for the Tensor cout print
static void print_recursive(
    std::ostream& os,
    Tensor& tensor,
    size_t dim_index,
    std::vector<size_t>& current_indices,
    int indent
) {
    size_t dim_size = tensor.m_shape[dim_index];
    
    if (dim_index == tensor.m_shape.size() - 1) {
        os << "[";
        for (size_t i = 0; i < dim_size; ++i) {
            current_indices.push_back(i);
            float val = tensor[current_indices];
            current_indices.pop_back();
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
            current_indices.push_back(i);
            print_recursive(
                os,
                tensor,
                dim_index + 1,
                current_indices,
                indent + 1
            );
            current_indices.pop_back();
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
        std::vector<size_t> current_indices;
        print_recursive(os, const_cast<Tensor&>(tensor), 0, current_indices, 12);
    }
    os << ")";
    return os;
}
