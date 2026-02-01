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

    virtual std::ostream& print(std::ostream& os) const;
    friend std::ostream& operator<<(std::ostream& os, const BackwardOp& op);

    virtual void backprop(Tensor& out) = 0;

    virtual void init_operands_grad_if_none() = 0;
    
    virtual void backprop_operands() = 0;
};

template <size_t N>
class NBackwardOp: public BackwardOp {
public:
    std::array<Tensor*, N> m_operands;

    NBackwardOp(
        std::array<Tensor*, N> operands
    ): m_operands(operands) {}

    void init_operands_grad_if_none() {
        for (size_t i {0}; i<N; i++) {
            Tensor* operand = m_operands[i];
            if (!operand->m_grad) {
                operand->m_grad = std::make_shared<Tensor>(operand->m_shape);
                operand->m_grad->m_requires_grad = false;
            }
        }
    }
    
    void backprop_operands() {
        for (size_t i {0}; i<N; i++) {
            m_operands[i]->backprop();
        }
    }
};

class BackwardMult : public NBackwardOp<2> {
public:
    static constexpr size_t N { 2 };
    
    using NBackwardOp<N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void backprop(Tensor& out) override;
};

class BackwardAdd : public NBackwardOp<2> {
public:
    static constexpr size_t N { 2 };

    using NBackwardOp<N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void backprop(Tensor& out) override;
};

class BackwardMinus : public NBackwardOp<1> {
public:
    static constexpr size_t N { 1 };

    using NBackwardOp<N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void backprop(Tensor& out) override;
};

class BackwardPow : public NBackwardOp<2> {
public:
    static constexpr size_t N { 2 };

    using NBackwardOp<N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void backprop(Tensor& out) override;
};

#endif
