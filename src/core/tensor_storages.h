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
    
    std::shared_ptr<std::vector<float>> m_flat_data;

    TensorStorage(
        std::vector<size_t> shape
    );
    
    friend std::ostream& operator<<(std::ostream& os, const TensorStorage& tensor);
    
    bool is_contiguous();

    float& get_entry_ref(
        size_t l_index
    );
    
    float& get_entry_ref(
        const std::vector<size_t>& md_index
    );
    
    void fill(
        float value
    );
    
    float& item();
    
    void slice(
        size_t dim,
        size_t slice_index
    );
    
    void dice(
        size_t dim,
        size_t index_start,
        size_t index_end
    );
    
    void linspace(
        float start,
        float end
    );
    
    void reshape(
        std::vector<size_t> shape
    );

    void transpose(
        size_t dim_1,
        size_t dim_2
    );

    TensorStorage clone() const;

    static bool are_shapes_equal(
        const TensorStorage& a,
        const TensorStorage& b
    );

    static TensorStorage s_mult(
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
    
private:
    static std::vector<size_t> s_init_strides(
        const std::vector<size_t>& shape
    );

    size_t get_flat_index_from_md(
        const std::vector<size_t>& md_index
    ) const;

    size_t get_flat_index_from_logical(
        size_t logical_index
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
