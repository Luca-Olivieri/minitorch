#ifndef TENSORS_H
#define TENSORS_H

#include <iostream>
#include <vector>
#include <mdspan>
#include <iomanip>
#include <string>
#include <memory>

// #include "ops.h"

class BackwardOp;

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector){
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

class Tensor {
public:
    std::vector<size_t> m_shape;
    size_t m_numel;
    std::vector<size_t> m_strides;
    std::vector<float> m_flat_data;
    size_t m_offset;
    
    std::shared_ptr<Tensor> m_grad;
    std::shared_ptr<BackwardOp> m_grad_fn;
    bool m_requires_grad;

    Tensor(
        std::vector<size_t> shape
    );

    ~Tensor();
    
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    
    size_t get_flat_index_from_md(
        const std::vector<size_t>& md_index
    ) const;

    size_t get_flat_index_from_logical(
        size_t logical_index
    ) const;

    bool is_contiguous();
    
    float& operator[](const std::vector<size_t>& md_index);
    
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

    static bool are_shapes_equal(
        const Tensor& a,
        const Tensor& b
    );

    Tensor operator*(
        const Tensor& other
    );
    
    Tensor operator+(
        const Tensor& other
    );
    
    Tensor& operator+=(
        const Tensor& other
    );
    
    Tensor operator-();
    
    Tensor pow(
        const Tensor& exp
    );

    void reset_grads();

    void backward();

    void backprop();
    
private:
    static std::vector<size_t> init_strides(
        const std::vector<size_t>& shape
    );

    float& get_entry_ref(
        size_t l_index
    );

    static Tensor mult(
        const Tensor& a,
        const Tensor& b
    );

    static Tensor add(
        const Tensor& a,
        const Tensor& b
    );
    
    static Tensor& add_inplace(
        Tensor& a,
        const Tensor& b
    );
    
    static Tensor minus(
        const Tensor& a
    );
    
    static Tensor s_pow(
        const Tensor& base,
        const Tensor& exp
    );

    template <typename BackwardOp, typename Func, typename... Tensors>
    static Tensor apply_op(Func op, Tensors&... operands) {
        // 1. Infer N at compile time
        constexpr size_t N = sizeof...(operands);
        static_assert(N == BackwardOp::N, "Number of tensors does not match BackwardOp arity");

        // 2. Get metadata from the first tensor (using a trick to access the first element of a pack)
        auto& first = [] (auto& head, [[maybe_unused]] auto&... tail) -> auto& { return head; }(operands...);
        size_t numel = first.m_numel;
        Tensor out(first.m_shape);

        // 3. Optional: Shape safety check using Fold Expressions
        if (!((operands.m_shape == first.m_shape) && ...)) {
            throw std::invalid_argument("Shapes must match for element-wise operation");
        }

        // 4. Computation loop
        for (size_t i = 0; i < numel; i++) {
            // The magic: "Unpack" the i-th element of every tensor into the lambda
            out.m_flat_data[i] = op(operands.m_flat_data[i]...);
        }

        // 5. Build the backward node
        if ((operands.m_requires_grad || ...)) {
            out.m_requires_grad = true;
            out.m_grad_fn = std::make_shared<BackwardOp>(std::array<Tensor*, N>{const_cast<Tensor*>(&operands)...});
        }
        
        return out;
    }
    
    template <typename BackwardOp, typename Func, typename... Tensors>
    static Tensor& apply_op_inplace(Func op, Tensors&... operands) {
        // 1. Infer N at compile time
        constexpr size_t N = sizeof...(operands);
        static_assert(N == BackwardOp::N, "Number of tensors does not match BackwardOp arity");

        // 2. Get metadata from the first tensor (using a trick to access the first element of a pack)
        auto& first = [] (auto& head, [[maybe_unused]] auto&... tail) -> auto& { return head; }(operands...);
        size_t numel = first.m_numel;
        Tensor& out = first;

        if (out.m_requires_grad) {
            throw std::logic_error(std::format("Cannot compute gradients on inplace operators. Set requires_grad = false."));
        }

        // 3. Optional: Shape safety check using Fold Expressions
        if (!((operands.m_shape == first.m_shape) && ...)) {
            throw std::invalid_argument("Shapes must match for element-wise operation");
        }

        // 4. Computation loop
        for (size_t i = 0; i < numel; i++) {
            // The magic: "Unpack" the i-th element of every tensor into the lambda
            out.m_flat_data[i] = op(operands.m_flat_data[i]...);
        }
        
        return out;
    }
};

#endif
