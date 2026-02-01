#include <iostream>
#include <format>
#include <math.h>
#include <memory>

#include "ops.h"
#include "tensors.h"

BackwardOp::BackwardOp() {}

std::ostream& BackwardOp::print(std::ostream& os) const {
    return os << "BackwardOp";
}

std::ostream& operator<<(std::ostream& os, const BackwardOp& op) {
    op.print(os);
    return os;
}

std::ostream& BackwardMult::print(std::ostream& os) const {
    return os << "BackwardMult";
}

void BackwardMult::backprop(Tensor& out) {
    for (size_t i {0}; i<N; i++) {
        if (auto operand = m_operands[i]) {
            operand->m_grad = std::make_shared<Tensor>(operand->m_shape);
        }
    }
    if (auto a = m_operands[0]) {
        *(a->m_grad) = out * out;
    }
    // if (!m_t1.m_grad) m_t1.m_grad = std::make_shared<Tensor>(m_t1.m_shape);
    // if (!m_t2.m_grad) m_t2.m_grad = std::make_shared<Tensor>(m_t2.m_shape);
    // *(m_t1.m_grad) = m_t1 * m_t2 * *(out.m_grad);
    // m_t1.backprop();
    // m_t2.backprop();
}

std::ostream& BackwardAdd::print(std::ostream& os) const {
    return os << "BackwardAdd";
}

void BackwardAdd::backprop(Tensor& out) {
    for (size_t i {0}; i<N; i++) {
        if (auto operand = m_operands[i]) {
            operand->m_grad = std::make_shared<Tensor>(operand->m_shape);
        }
    }
    if (auto a = m_operands[0]) {
        *(a->m_grad) = out * out;
    }
}
