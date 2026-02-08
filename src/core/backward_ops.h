#ifndef BACKWARD_OPS
#define BACKWARD_OPS

#include <iostream>
#include <memory>
#include <vector>
#include <functional>

#include "tensor_nodes.h"

class BackwardOp {
public:
    BackwardOp();

    virtual ~BackwardOp() = default;

    virtual std::ostream& print(std::ostream& os) const = 0;
    friend std::ostream& operator<<(std::ostream& os, const BackwardOp& op);

    // virtual void init_operands_grad_if_none() = 0;
    
    virtual void reset_all_grads() = 0;
    
    virtual void compute_operands_grad(const Tensor& out, bool create_graph = false) = 0;

    virtual std::vector<Tensor> get_operands() const = 0;
};

template <size_t N>
class NBackwardOp: public BackwardOp {
public:
    static constexpr size_t s_N = N; // expose N as a static member
    std::array<Tensor, N> m_operands;

        // 1. Variadic Template Constructor
    template <typename... Tensors>
    explicit NBackwardOp(
        Tensors... operands
    ): m_operands{{operands...}} { // Optimization note: Move semantics (Tensors&&... and std::forward) could be used here for efficiency to avoid atomic ref-count increments on shared_ptr
        // 2. Compile-time Arity Check
        static_assert(sizeof...(Tensors) == N, 
            "Error: Number of arguments provided to constructor must match template parameter N.");
            
        // 3. (Optional) Strict Type Checking using C++17 Fold Expressions
        // This ensures every argument is actually a Tensor
        static_assert((std::is_convertible_v<Tensors, Tensor> && ...), 
            "Error: All arguments must be implicitly convertible to Tensor.");
    }

    /*
    void init_operands_grad_if_none() {
        for (size_t i {0}; i<N; i++) {
            Tensor& operand = m_operands[i];

            if (!operand.m_node->m_grad) {
                operand.m_node->m_grad = std::make_shared<TensorNode>(operand.shape());
                operand.m_node->m_grad->m_requires_grad = false;
            }
        }
    }
    */

    void reset_all_grads() override {
        for (size_t i {0}; i<N; i++) {
            Tensor& operand = m_operands[i];
            operand.zero_grad();
        }
    }

    std::vector<Tensor> get_operands() const override {
        std::vector<Tensor> ops;
        ops.reserve(N);
        for(const auto& op : m_operands) {
            ops.push_back(op);
        }
        return ops;
    }
};

class BackwardView: public NBackwardOp<1> {
public:
    using NBackwardOp<s_N>::NBackwardOp;
};

class BackwardReshape : public BackwardView {
public:
    using BackwardView::BackwardView;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardTranspose : public BackwardView {
public:
    size_t m_dim_1;
    size_t m_dim_2;

    BackwardTranspose(
        Tensor viewed_tensor,
        size_t dim_1,
        size_t dim_2
    );

    // using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardAdd : public NBackwardOp<2> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardMinus : public NBackwardOp<1> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardSub : public NBackwardOp<2> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardMult : public NBackwardOp<2> {
public:    
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardPow : public NBackwardOp<2> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardLog : public NBackwardOp<1> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};


#endif
