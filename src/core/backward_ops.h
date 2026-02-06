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

    virtual void init_operands_grad_if_none() = 0;
    
    virtual void reset_all_grads() = 0;
    
    virtual void compute_operands_grad(TensorNode& out) = 0;
    
    virtual void backprop() = 0;
};

template <size_t N>
class NBackwardOp: public BackwardOp {
public:
    static constexpr size_t s_N = N; // expose N as a static member
    std::array<TensorNode*, N> m_operands;

        // 1. Variadic Template Constructor
    template <typename... Tensors>
    explicit NBackwardOp(Tensors... operands): m_operands{{operands...}} {
        // 2. Compile-time Arity Check
        static_assert(sizeof...(Tensors) == N, 
            "Error: Number of arguments provided to constructor must match template parameter N.");
            
        // 3. (Optional) Strict Type Checking using C++17 Fold Expressions
        // This ensures every argument is actually a TensorNode*
        static_assert((std::is_convertible_v<Tensors, TensorNode*> && ...), 
            "Error: All arguments must be implicitly convertible to TensorNode*.");
    }

    void init_operands_grad_if_none() {
        for (size_t i {0}; i<N; i++) {
            TensorNode* operand = m_operands[i];

            if (!operand->m_grad) {
                operand->m_grad = std::make_shared<TensorNode>(operand->m_value.m_shape);
                operand->m_grad->m_requires_grad = false;
            }
        }
    }

    void backprop() {
        for (size_t i {0}; i<N; i++) {
            TensorNode* operand = m_operands[i];
            operand->backprop();
        }
    }

    void reset_all_grads() {
        for (size_t i {0}; i<N; i++) {
            TensorNode* operand = m_operands[i];
            operand->zero_grad();
        }
    }
};

class BackwardAdd : public NBackwardOp<2> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(TensorNode& out) override;
};

class BackwardMinus : public NBackwardOp<1> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(TensorNode& out) override;
};

class BackwardMult : public NBackwardOp<2> {
public:    
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(TensorNode& out) override;
};

class BackwardPow : public NBackwardOp<2> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(TensorNode& out) override;
};

#endif
