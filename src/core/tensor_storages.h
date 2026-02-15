#ifndef TENSOR_STORAGES_H
#define TENSOR_STORAGES_H

#include <iostream>
#include <vector>
#include <mdspan>
#include <iomanip>
#include <string>
#include <memory>

class TensorStorage {
public:
    std::vector<size_t> m_shape;
    size_t m_numel;
    std::vector<size_t> m_strides;
    size_t m_offset;
    
    std::unique_ptr<std::vector<float>> m_flat_data;

    TensorStorage(
        std::vector<size_t> shape,
        float value = 0.0f
    );

    static TensorStorage linspace(
        std::vector<size_t> shape,
        float start,
        float end
    );
    
    friend std::ostream& operator<<(std::ostream& os, const TensorStorage& tensor);
    
    bool is_contiguous() const;

    float& get_entry_ref(
        size_t l_index
    );
    
    float& get_entry_ref(
        const std::vector<size_t>& md_index
    );
    
    TensorStorage fill(
        float value
    );

    void fill_inplace(
        float value
    );
    
    TensorStorage linspace(
        float start,
        float end
    );

    void linspace_inplace(
        float start,
        float end
    );

    float& item();

    TensorStorage clone() const;

    static bool are_shapes_equal(
        const TensorStorage& a,
        const TensorStorage& b
    );

    static TensorStorage s_mult(
        const TensorStorage& a,
        const TensorStorage& b
    );

    static TensorStorage s_div(
        const TensorStorage& a,
        const TensorStorage& b
    );

    static TensorStorage s_add(
        const TensorStorage& a,
        const TensorStorage& b
    );
    
    static TensorStorage& s_add_inplace(
        TensorStorage& a,
        const TensorStorage& b
    );

    static TensorStorage s_sub(
        const TensorStorage& a,
        const TensorStorage& b
    );
    
    static TensorStorage s_minus(
        const TensorStorage& a
    );
    
    static TensorStorage s_pow(
        const TensorStorage& base,
        const TensorStorage& exp
    );
    
    static TensorStorage s_log(
        const TensorStorage& arg
    );

    static std::vector<size_t> reduce_shape(
        const std::vector<size_t>& shape,
        const size_t dim
    );
    
    static std::vector<size_t> unsqueeze_shape(
        const std::vector<size_t>& shape,
        const size_t dim
    );

    static void populate_in_md_for_accum(
        std::vector<size_t>& in_md,
        const std::vector<size_t>& out_md,
        const size_t dim
    );

    static TensorStorage s_sum(
        TensorStorage& a,
        const size_t dim
    );
    
    static TensorStorage s_unsqueeze(
        TensorStorage& a,
        const size_t dim
    );
    
    static TensorStorage s_squeeze(
        TensorStorage& a,
        const size_t dim
    );
    
    static TensorStorage s_repeat(
        TensorStorage& a,
        const size_t dim,
        const size_t times
    );
    
private:
    TensorStorage(
        std::vector<size_t> shape,
        float start,
        float end
    );

    static void assert_positive_dims(
        const std::vector<size_t>& shape
    );
    
    static size_t compute_numel_from_shape(
        const std::vector<size_t>& shape
    );

    static std::vector<size_t> s_init_strides(
        const std::vector<size_t>& shape
    );

    size_t md_to_flat(
        const std::vector<size_t>& md_index
    ) const;

    size_t logical_to_flat(
        size_t logical_index
    ) const;

    std::vector<size_t> logical_to_md(
        size_t l_index
    ) const;

    template <typename Func, typename... Tensors>
    static TensorStorage s_apply_op(Func op, Tensors&... operands) {
        // 1. Infer N at compile time
        // constexpr size_t N = sizeof...(operands);
        // static_assert(N == BackwardOp::N, "Number of tensors does not match BackwardOp arity");

        // 2. Get metadata from the first tensor (using a trick to access the first element of a pack)
        auto& first = [] (auto& head, [[maybe_unused]] auto&... tail) -> auto& { return head; }(operands...);
        size_t numel = first.m_numel;
        TensorStorage out(first.m_shape);

        // 3. Optional: Shape safety check using Fold Expressions
        if (!((operands.m_shape == first.m_shape) && ...)) {
            throw std::invalid_argument("Shapes must match for element-wise operation");
        }

        // 4. Computation loop
        for (size_t i = 0; i < numel; i++) {
            // The magic: "Unpack" the i-th element of every tensor into the lambda
            (*out.m_flat_data)[i] = op((*operands.m_flat_data)[i]...);
        }

        return out;
    }
    
    template <typename Func, typename... Tensors>
    static TensorStorage& s_apply_op_inplace(Func op, Tensors&... operands) {
        // 1. Infer N at compile time
        // constexpr size_t N = sizeof...(operands);
        // static_assert(N == BackwardOp::N, "Number of tensors does not match BackwardOp arity");

        // 2. Get metadata from the first tensor (using a trick to access the first element of a pack)
        auto& first = [] (auto& head, [[maybe_unused]] auto&... tail) -> auto& { return head; }(operands...);
        size_t numel = first.m_numel;
        TensorStorage& out = first;

        // if (out.m_requires_grad) {
        //     throw std::logic_error(std::format("Cannot compute gradients on inplace operators. Set requires_grad = false."));
        // }

        // 3. Optional: Shape safety check using Fold Expressions
        if (!((operands.m_shape == first.m_shape) && ...)) {
            throw std::invalid_argument("Shapes must match for element-wise operation");
        }

        // 4. Computation loop
        for (size_t i = 0; i < numel; i++) {
            // The magic: "Unpack" the i-th element of every tensor into the lambda
            (*out.m_flat_data)[i] = op((*operands.m_flat_data)[i]...);
        }
        
        return out;
    }
};

#endif
