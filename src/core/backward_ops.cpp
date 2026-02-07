#include <iostream>
#include <format>
#include <math.h>
#include <memory>

#include "backward_ops.h"

#include "formatting.h"

#include "tensor_nodes.h"
#include "tensors.h"

BackwardOp::BackwardOp() {}

std::ostream& operator<<(std::ostream& os, const BackwardOp& op) {
    return op.print(os);
}

std::ostream& BackwardAdd::print(std::ostream& os) const {
    return os << "BackwardAdd";
}

void BackwardAdd::compute_operands_grad(TensorNode& out) {
    TensorNode& a = *m_operands[0];
    TensorNode& b = *m_operands[1];
    *(a.m_grad) += *(out.m_grad);
    *(b.m_grad) += *(out.m_grad);
}

std::ostream& BackwardMinus::print(std::ostream& os) const {
    return os << "BackwardMinus";
}

void BackwardMinus::compute_operands_grad(TensorNode& out) {
    TensorNode& a = *m_operands[0];
    *(a.m_grad) += *(-*(out.m_grad));
}

std::ostream& BackwardSub::print(std::ostream& os) const {
    return os << "BackwardSub";
}

void BackwardSub::compute_operands_grad(TensorNode& out) {
    TensorNode& a = *m_operands[0];
    TensorNode& b = *m_operands[1];
    *(a.m_grad) += *(out.m_grad);
    *(b.m_grad) += *(-*(out.m_grad));
}

std::ostream& BackwardMult::print(std::ostream& os) const {
    return os << "BackwardMult";
}

void BackwardMult::compute_operands_grad(TensorNode& out) {
    TensorNode& a = *m_operands[0];
    TensorNode& b = *m_operands[1];
    *(a.m_grad) += *(b * *(out.m_grad));
    *(b.m_grad) += *(a * *(out.m_grad));
}

std::ostream& BackwardPow::print(std::ostream& os) const {
    return os << "BackwardPow";
}

void BackwardPow::compute_operands_grad(TensorNode& out) {
    TensorNode& base = *m_operands[0];
    TensorNode& exp = *m_operands[1];
    std::shared_ptr<TensorNode> ones = std::make_shared<TensorNode>(exp.m_value.m_shape);
    ones->m_value.fill(1.0f);
    *(base.m_grad) += *(*(exp * *base.pow(*(exp + *(-(*ones))))) * *(out.m_grad));
    *(exp.m_grad) += *(*(*base.pow(exp) * *(base.log())) * *(out.m_grad));
}

std::ostream& BackwardLog::print(std::ostream& os) const {
    return os << "BackwardPow";
}

void BackwardLog::compute_operands_grad([[maybe_unused]] TensorNode& out) {
    // TensorNode& arg = *m_operands[0];
}
