#include <iostream>
#include <format>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <cmath>

#include "tensor_storages.h"
#include "formatting.h"

TensorStorage::TensorStorage(
    std::vector<size_t> shape
): m_offset(0) {

    assert_positive_dims(shape);

    size_t numel { compute_numel_from_shape(shape) };
    
    m_numel = numel;
    m_shape = shape;
    m_strides = TensorStorage::s_init_strides(m_shape);
    
    m_flat_data = std::make_unique<std::vector<float>>(m_numel, 0.0f);
}

void TensorStorage::assert_positive_dims(
    const std::vector<size_t>& shape
) {
    for (size_t dim : shape) {
        if (dim == 0) {
            throw std::invalid_argument(std::format("\nTensor shape must have positive dimensions. Got {}.", shape));
        }
    }
}

size_t TensorStorage::compute_numel_from_shape(
    const std::vector<size_t>& shape
) {
    size_t numel { 1 };
    for (size_t dim : shape) numel *= dim;
    return numel;
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

bool TensorStorage::is_contiguous() const {
    std::vector<size_t> contiguous_strides = TensorStorage::s_init_strides(m_shape);
    for (size_t i = 0; i < m_shape.size(); ++i) {
        if (m_shape[i] == 1) continue; // singleton dimensions do not affect contiguity
        if (m_strides[i] != contiguous_strides[i]) return false;
    }
    return true;
}

void TensorStorage::fill_inplace(
    float value
) {
    for (size_t i = 0; i < m_numel; i++) {
        get_entry_ref(i) = value;
    }
}

void TensorStorage::linspace_inplace(
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

TensorStorage TensorStorage::clone() const {
    TensorStorage out(m_shape);
    out.m_offset = m_offset;
    out.m_strides = m_strides;
    *out.m_flat_data = *m_flat_data;
    out.m_numel = m_numel;
    return out;
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

TensorStorage TensorStorage::s_sub(const TensorStorage& a, const TensorStorage& b) {
    return TensorStorage::s_add(a, TensorStorage::s_minus(b));
}

TensorStorage TensorStorage::s_pow(const TensorStorage& base, const TensorStorage& exp) {
    return s_apply_op(
        [](float base_, float exp_) { return std::pow(base_, exp_); }, 
        base, exp
    );
}

TensorStorage TensorStorage::s_log(const TensorStorage& arg) {
    return s_apply_op(
        [](float arg_) { return std::log(arg_); }, 
        arg
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
