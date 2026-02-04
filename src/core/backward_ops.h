#ifndef OPS_H
#define OPS_H

#include <iostream>
#include <memory>
#include <vector>
#include <functional>

#include "tensors.h"

class BackwardOp {
public:
    BackwardOp();

    virtual ~BackwardOp() = default;

    virtual std::ostream& print(std::ostream& os) const = 0;
    friend std::ostream& operator<<(std::ostream& os, const BackwardOp& op);

    virtual void backprop(Tensor& out) = 0;

    virtual void init_operands_grad_if_none() = 0;
    
    virtual void backprop_operands() = 0;
};

template <size_t N>
class NBackwardOp: public BackwardOp {
public:
    static constexpr size_t s_N = N; // expose N as a static member
    std::array<Tensor*, N> m_operands;

        // 1. Variadic Template Constructor
    template <typename... Tensors>
    explicit NBackwardOp(Tensors... operands): m_operands{{operands...}} {
        // 2. Compile-time Arity Check
        static_assert(sizeof...(Tensors) == N, 
            "Error: Number of arguments provided to constructor must match template parameter N.");
            
        // 3. (Optional) Strict Type Checking using C++17 Fold Expressions
        // This ensures every argument is actually a Tensor*
        static_assert((std::is_convertible_v<Tensors, Tensor*> && ...), 
            "Error: All arguments must be implicitly convertible to Tensor*.");
    }

    void init_operands_grad_if_none() {
        for (size_t i {0}; i<N; i++) {
            Tensor* operand = m_operands[i];

            if (!operand->m_grad) {
                operand->m_grad = std::make_shared<Tensor>(operand->m_value.m_shape);
                operand->m_grad->m_requires_grad = false;
            }
        }
    }
    
    void backprop_operands() {
        // for (size_t i {0}; i<N; i++) {
        //     m_operands[i]->backprop();
        // }
    }
};

class BackwardAdd : public NBackwardOp<2> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void backprop(Tensor& out) override;
};

class BackwardMinus : public NBackwardOp<1> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void backprop(Tensor& out) override;
};

class BackwardMult : public NBackwardOp<2> {
public:    
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void backprop(Tensor& out) override;
};

class BackwardPow : public NBackwardOp<2> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void backprop(Tensor& out) override;
};

#endif
