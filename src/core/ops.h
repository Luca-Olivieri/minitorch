#ifndef OPS_H
#define OPS_H

#include <iostream>
#include <memory>
#include <vector>
#include <functional>

class Tensor;

class BackwardOp {
public:
    BackwardOp();

    virtual ~BackwardOp() = default;

    virtual std::ostream& print(std::ostream& os) const;
    friend std::ostream& operator<<(std::ostream& os, const BackwardOp& op);

    virtual void backprop(Tensor& out) = 0;
};

template <size_t N>
class NBackwardOp: public BackwardOp {
public:
    std::array<Tensor*, N> m_operands;

    NBackwardOp(
        std::array<Tensor*, N> operands
    ): m_operands(operands) {}
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

#endif
