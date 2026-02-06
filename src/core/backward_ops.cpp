#include <iostream>
#include <format>
#include <math.h>
#include <memory>

#include "backward_ops.h"

#include "tensors.h"

BackwardOp::BackwardOp() {}

std::ostream& operator<<(std::ostream& os, const BackwardOp& op) {
    return op.print(os);
}

std::ostream& BackwardAdd::print(std::ostream& os) const {
    return os << "BackwardAdd";
}

void BackwardAdd::compute_operands_grad(Tensor& out) {
    Tensor& a = *m_operands[0];
    Tensor& b = *m_operands[1];
    *(a.m_grad) += *(out.m_grad);
    *(b.m_grad) += *(out.m_grad);
}

std::ostream& BackwardMinus::print(std::ostream& os) const {
    return os << "BackwardMinus";
}

void BackwardMinus::compute_operands_grad(Tensor& out) {
    Tensor& a = *m_operands[0];
    *(a.m_grad) += -*(out.m_grad);
}

std::ostream& BackwardMult::print(std::ostream& os) const {
    return os << "BackwardMult";
}

void BackwardMult::compute_operands_grad(Tensor& out) {
    Tensor& a = *m_operands[0];
    Tensor& b = *m_operands[1];
    *(a.m_grad) += b * *(out.m_grad);
    *(b.m_grad) += a * *(out.m_grad);
}

std::ostream& BackwardPow::print(std::ostream& os) const {
    return os << "BackwardPow";
}

void BackwardPow::compute_operands_grad(Tensor& out) {
    Tensor& base = *m_operands[0];
    Tensor& exp = *m_operands[1];
    Tensor ones{exp.m_value.m_shape};
    ones.fill(1.0f);
    *(base.m_grad) += exp * base.pow(exp+(-ones)) * *(out.m_grad);
    // *(b.m_grad) += *(out.m_grad);
}
