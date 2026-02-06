#include <iostream>
#include <format>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <cmath>

#include "tensors_storage.h"
#include "formatting.h"

TensorStorage::TensorStorage(
    std::vector<size_t> shape
): 
    m_shape(shape),
    m_offset(0) {
        
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
    m_strides = TensorStorage::s_init_strides(m_shape);
    m_flat_data = std::make_shared<std::vector<float>>(m_numel, 0.0f);
    
}

std::vector<size_t> TensorStorage::s_init_strides(
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

size_t TensorStorage::get_flat_index_from_md(
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

size_t TensorStorage::get_flat_index_from_logical(
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

float& TensorStorage::get_entry_ref(
    size_t l_index
) {
    return (*m_flat_data)[get_flat_index_from_logical(l_index)];
}

float& TensorStorage::get_entry_ref(
    const std::vector<size_t>& md_index
) {
    // Scalar case
    if (m_shape.empty()) {
        throw std::invalid_argument(std::format("\nScalar tensor cannot be access by index. Got index {}", md_index));
    }
    return (*m_flat_data)[get_flat_index_from_md(md_index)];
}

bool TensorStorage::is_contiguous() {
    std::vector<size_t> contiguous_strides = TensorStorage::s_init_strides(m_shape);
    for (size_t i = 0; i < m_shape.size(); ++i) {
        if (m_shape[i] == 1) continue; // singleton dimensions do not affect contiguity
        if (m_strides[i] != contiguous_strides[i]) return false;
    }
    return true;
}

void TensorStorage::fill(
    float value
) {
    for (size_t i = 0; i < m_numel; i++) {
        get_entry_ref(i) = value;
    }
}

void TensorStorage::linspace(
    float start,
    float end
) {
    float delta = (end-start)/(static_cast<float>(m_numel-1));
    for (size_t i = 0; i < m_numel; i++) {
        get_entry_ref(i) = start + static_cast<float>(i)*delta;
    }
}

float& TensorStorage::item() {
    if (m_numel != 1) {
        throw std::runtime_error(std::format("Cannot call item() on a non-singleton tensor (shape {}).", m_shape));
    }
    return (*m_flat_data)[m_offset];
}

void TensorStorage::slice(
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

void TensorStorage::dice(
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

void TensorStorage::reshape(
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
    m_strides = TensorStorage::s_init_strides(m_shape);
}

void TensorStorage::transpose(
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

TensorStorage TensorStorage::clone() const {
    TensorStorage cloned(m_shape);
    cloned.m_offset = m_offset;
    cloned.m_strides = m_strides;
    cloned.m_flat_data = m_flat_data;
    cloned.m_numel = m_numel;
    return cloned;
}

bool TensorStorage::are_shapes_equal(
    const TensorStorage& a,
    const TensorStorage& b
) {
    return a.m_shape == b.m_shape;
}

TensorStorage TensorStorage::s_mult(const TensorStorage& a, const TensorStorage& b) {
    return s_apply_op(
        [](float x, float y) { return x * y; }, 
        a, b
    );
}

TensorStorage TensorStorage::s_add(const TensorStorage& a, const TensorStorage& b) {
    return s_apply_op(
        [](float x, float y) { return x + y; }, 
        a, b
    );
}

TensorStorage& TensorStorage::s_add_inplace(TensorStorage& a, const TensorStorage& b) {
    return s_apply_op_inplace(
        [](float x, float y) { return x + y; }, 
        a, b
    );
}

TensorStorage TensorStorage::s_minus(const TensorStorage& a) {
    return s_apply_op(
        [](float x) { return -x; }, 
        a
    );
}

TensorStorage TensorStorage::s_pow(const TensorStorage& base, const TensorStorage& exp) {
    return s_apply_op(
        [](float base, float exp) { return std::pow(base, exp); }, 
        base, exp
    );
}

static void s_print_recursive(
    std::ostream& os,
    TensorStorage& tensor_impl,
    size_t dim_index,
    std::vector<size_t>& current_indices,
    int indent
) {
    size_t dim_size = tensor_impl.m_shape[dim_index];
    
    if (dim_index == tensor_impl.m_shape.size() - 1) {
        os << "[";
        for (size_t i = 0; i < dim_size; ++i) {
            current_indices.push_back(i);
            float val = tensor_impl.get_entry_ref(current_indices);
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
                size_t newlines = tensor_impl.m_shape.size() - dim_index - 1;
                for (size_t nl = 0; nl < newlines; ++nl) os << "\n";
                for (int k = 0; k < indent + 1; ++k) os << " ";
            }
            current_indices.push_back(i);
            s_print_recursive(
                os,
                tensor_impl,
                dim_index + 1,
                current_indices,
                indent + 1
            );
            current_indices.pop_back();
        }
        os << "]";
    }
}

std::ostream& operator<<(std::ostream& os, const TensorStorage& tensor_impl){
    os << std::format("Tensor(shape={}, dtype=float,\n       data=", tensor_impl.m_shape);
    if (tensor_impl.m_shape.empty()) {
        if (!tensor_impl.m_flat_data->empty())
             os << std::fixed << std::setprecision(4) << (*tensor_impl.m_flat_data)[0];
    } else {
        std::vector<size_t> current_indices;
        s_print_recursive(os, const_cast<TensorStorage&>(tensor_impl), 0, current_indices, 12);
    }
    os << ")";
    return os;
}
