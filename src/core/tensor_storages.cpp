#include <iostream>
#include <format>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <cmath>

#include "tensor_storages.h"
#include "formatting.h"

TensorStorage::TensorStorage(
        const std::vector<size_t>& shape,
        const float value
): m_offset(0) {

    assert_positive_dims(shape);

    size_t numel { compute_numel_from_shape(shape) };
    
    m_numel = numel;
    m_shape = shape;
    m_strides = TensorStorage::s_init_strides(m_shape);
    
    m_flat_data = std::make_unique<std::vector<float>>(
        m_numel,
        value
    );
}

TensorStorage::TensorStorage(
        const std::vector<size_t>& shape,
        const float start,
        const float end
): m_offset(0) {

    assert_positive_dims(shape);

    size_t numel { compute_numel_from_shape(shape) };
    
    m_numel = numel;
    m_shape = shape;
    m_strides = TensorStorage::s_init_strides(m_shape);
    
    m_flat_data = std::make_unique<std::vector<float>>();
    m_flat_data->reserve(m_numel);
    linspace_inplace(start, end);
}

TensorStorage TensorStorage::linspace(
    const std::vector<size_t>& shape,
    const float start,
    const float end
) {
    return TensorStorage(shape, start, end);
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

size_t TensorStorage::md_to_flat(
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

size_t TensorStorage::logical_to_flat(
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

std::vector<size_t> TensorStorage::logical_to_md(
    size_t l_index
) const {
    if (l_index >= m_numel) {
        throw std::out_of_range(
            std::format(
                "Logical index {} out of bounds for tensor of size {}.",
                l_index, m_numel)
        );
    }

    std::vector<size_t> md(m_shape.size());

    for (size_t i = m_shape.size(); i-- > 0;) {
        md[i] = l_index % m_shape[i];
        l_index /= m_shape[i];
    }

    return md;
}

float& TensorStorage::get_entry_ref(
    const size_t l_index
) const {
    return (*m_flat_data)[logical_to_flat(l_index)];
}

float& TensorStorage::get_entry_ref(
    const std::vector<size_t>& md_index
) const {
    // Scalar case
    if (m_shape.empty()) {
        throw std::invalid_argument(std::format("\nScalar tensor cannot be access by index. Got index {}", md_index));
    }
    return (*m_flat_data)[md_to_flat(md_index)];
}

bool TensorStorage::is_contiguous() const {
    std::vector<size_t> contiguous_strides = TensorStorage::s_init_strides(m_shape);
    for (size_t i = 0; i < m_shape.size(); ++i) {
        if (m_shape[i] == 1) continue; // singleton dimensions do not affect contiguity
        if (m_strides[i] != contiguous_strides[i]) return false;
    }
    return true;
}

TensorStorage TensorStorage::fill(
    const float value
) const {
    return TensorStorage(m_shape, value);
}

void TensorStorage::fill_inplace(
    const float value
) {
    for (size_t i = 0; i < m_numel; i++) {
        get_entry_ref(i) = value;
    }
}

TensorStorage TensorStorage::linspace(
    const float start,
    const float end
) const {
    TensorStorage out { m_shape };
    out.linspace_inplace(start, end);
    return out;
}

void TensorStorage::linspace_inplace(
    const float start,
    const float end
) {
    float delta = (end-start)/(static_cast<float>(m_numel-1));
    for (size_t i = 0; i < m_numel; i++) {
        get_entry_ref(i) = start + static_cast<float>(i)*delta;
    }
}

float& TensorStorage::item() const {
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

TensorStorage TensorStorage::s_mult(
        const TensorStorage& a, 
        const TensorStorage& b
) {
    return s_apply_op(
        [](float x, float y) { return x * y; }, 
        a, b
    );
}

TensorStorage TensorStorage::s_div(
        const TensorStorage& a,
        const TensorStorage& b
) {
    return s_apply_op(
        [](float x, float y) { return x / y; }, 
        a, b
    );
}

TensorStorage TensorStorage::s_add(
        const TensorStorage& a,
        const TensorStorage& b
) {
    return s_apply_op(
        [](float x, float y) { return x + y; }, 
        a, b
    );
}

TensorStorage& TensorStorage::s_add_inplace(
        TensorStorage& a,
        const TensorStorage& b
) {
    return s_apply_op_inplace(
        [](float x, float y) { return x + y; }, 
        a, b
    );
}

TensorStorage TensorStorage::s_minus(
        const TensorStorage& a
) {
    return s_apply_op(
        [](float x) { return -x; }, 
        a
    );
}

TensorStorage TensorStorage::s_sub(
        const TensorStorage& a,
        const TensorStorage& b
) {
    return TensorStorage::s_add(a, TensorStorage::s_minus(b));
}

TensorStorage TensorStorage::s_pow(const TensorStorage& base, const TensorStorage& exp) {
    return s_apply_op(
        [](float base_, float exp_) { return std::pow(base_, exp_); }, 
        base, exp
    );
}

TensorStorage TensorStorage::s_log(
        const TensorStorage& arg
) {
    return s_apply_op(
        [](float arg_) { return std::log(arg_); }, 
        arg
    );
}

 std::vector<size_t> TensorStorage::reduce_shape(
        const std::vector<size_t>& shape,
        const size_t dim
) {
    std::vector<size_t> out_shape = shape;
    
    out_shape.erase(
        out_shape.begin() +
        static_cast<std::vector<size_t>::difference_type>(dim)
    );

    return out_shape;
}

std::vector<size_t> TensorStorage::unsqueeze_shape(
        const std::vector<size_t>& shape,
        const size_t dim
) {
    std::vector<size_t> out_shape = shape;

    out_shape.insert(
        out_shape.begin() + static_cast<std::vector<size_t>::difference_type>(dim),
         1
    ); // inserts 1 at index 'dim'

    return out_shape;
}

void TensorStorage::populate_in_md_for_accum(
        std::vector<size_t>& in_md,
        const std::vector<size_t>& out_md,
        const size_t dim
) {
    // copy output coords into input coords (skip reduced dim)
    for (size_t i = 0, j = 0; i < in_md.size(); ++i) {
        if (i == dim) continue;
        in_md[i] = out_md[j++];
    }
}

TensorStorage TensorStorage::s_sum(
        const TensorStorage& a,
        const size_t dim
) {
    if (dim >= a.m_shape.size()) {
        throw std::invalid_argument(
            std::format("Reduction dimension {} out of range for shape {}.",
            dim, a.m_shape
            )
        );
    }

    // build output shape
    std::vector<size_t> out_shape = reduce_shape(a.m_shape, dim);

    TensorStorage out{ out_shape };

    // build input md index with dim inserted
    std::vector<size_t> in_md(a.m_shape.size());

    // iterate over output logical indices
    for (size_t out_i = 0; out_i < out.m_numel; ++out_i) {

        std::vector<size_t> out_md = out.logical_to_md(out_i);

        // in_md follows out_md but skips reduced dimension
        populate_in_md_for_accum(in_md, out_md, dim);

        // accumulate over reduced dim
        float acc = 0.0f;
        for (size_t r = 0; r < a.m_shape[dim]; ++r) {
            in_md[dim] = r;
            acc += a.get_entry_ref(in_md);
        }

        out.get_entry_ref(out_i) = acc;
    }

    return out;
}

TensorStorage TensorStorage::s_unsqueeze(
        const TensorStorage& a,
        const size_t dim
) {
    if (dim > a.m_shape.size()) {
        throw std::invalid_argument(
            std::format("Unsqueezed dimension {} out of range for shape of length {}.",
            dim, a.m_shape.size()
            )
        );
    }

    // build output shape
    std::vector<size_t> out_shape = unsqueeze_shape(a.m_shape, dim);

    TensorStorage out{ out_shape };

    // iterate over output logical indices and map to input multi-dim coords
    std::vector<size_t> in_md(a.m_shape.size());
    for (size_t out_i = 0; out_i < out.m_numel; ++out_i) {
        // Special-case scalar input: read by logical index 0
        if (a.m_shape.empty()) {
            out.get_entry_ref(out_i) = a.get_entry_ref(0);
            continue;
        }

        std::vector<size_t> out_md = out.logical_to_md(out_i);

        // map out_md -> in_md by skipping the inserted singleton dim
        size_t j = 0;
        for (size_t i = 0; i < out_md.size(); ++i) {
            if (i == dim) continue; // skip the singleton
            in_md[j++] = out_md[i];
        }

        out.get_entry_ref(out_i) = a.get_entry_ref(in_md);
    }

    return out;
}

TensorStorage TensorStorage::s_repeat(
        const TensorStorage& a,
        const size_t dim,
        const size_t times
) {
    if (dim >= a.m_shape.size()) {
        throw std::invalid_argument(
            std::format("Expanded dimension {} out of range for shape of length {}.",
            dim, a.m_shape.size()
            )
        );
    }
    if (a.m_shape[dim] != 1) {
        throw std::invalid_argument(
            std::format("Expanded dimension {} must be singleton. Got shqpe {}.",
            dim, a.m_shape.size()
            )
        );
    }

    // build output shape
    std::vector<size_t> out_shape = a.m_shape;
    out_shape[dim] = times;

    TensorStorage out{ out_shape };

    // iterate over output logical indices and map to input multi-dim coords
    std::vector<size_t> in_md(a.m_shape.size());
    for (size_t out_i = 0; out_i < out.m_numel; ++out_i) {
        std::vector<size_t> out_md = out.logical_to_md(out_i);

        // input coord equals output coord except at repeated dim which maps to 0
        for (size_t i = 0; i < out_md.size(); ++i) {
            in_md[i] = out_md[i];
        }
        in_md[dim] = 0;

        out.get_entry_ref(out_i) = a.get_entry_ref(in_md);
    }

    return out;
}

TensorStorage TensorStorage::s_squeeze(
        const TensorStorage& a,
        const size_t dim
) {
    if (dim >= a.m_shape.size()) {
        throw std::invalid_argument(
            std::format("Squeezed dimension {} out of range for shape of length {}.",
            dim, a.m_shape.size()
            )
        );
    }
    if (a.m_shape[dim] != 1) {
        throw std::invalid_argument(
            std::format("Squeezed dimension {} must be singleton. Got size {}.", dim, a.m_shape[dim])
        );
    }

    // build output shape
    std::vector<size_t> out_shape = reduce_shape(a.m_shape, dim);

    TensorStorage out{ out_shape };

    // iterate over output logical indices and map to input multi-dim coords
    std::vector<size_t> in_md(a.m_shape.size());
    for (size_t out_i = 0; out_i < out.m_numel; ++out_i) {
        // Special-case scalar input (shouldn't normally happen here, but be defensive)
        if (a.m_shape.empty()) {
            out.get_entry_ref(out_i) = a.get_entry_ref(0);
            continue;
        }

        std::vector<size_t> out_md = out.logical_to_md(out_i);

        // map out_md -> in_md by inserting the singleton coord (0) at position dim
        size_t j = 0;
        for (size_t i = 0; i < a.m_shape.size(); ++i) {
            if (i == dim) {
                in_md[i] = 0;
            } else {
                in_md[i] = out_md[j++];
            }
        }

        out.get_entry_ref(out_i) = a.get_entry_ref(in_md);
    }

    return out;
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
