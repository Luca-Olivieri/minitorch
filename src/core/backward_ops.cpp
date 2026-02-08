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

std::ostream& BackwardReshape::print(std::ostream& os) const {
    return os << "BackwardReshape";
}

void BackwardReshape::compute_operands_grad(const Tensor& out) {
    std::vector<size_t> original_shape = m_operands[1].m_node->m_storage.m_shape;
    m_operands[0].grad() += out.grad().reshape(original_shape);
}

std::ostream& BackwardAdd::print(std::ostream& os) const {
    return os << "BackwardAdd";
}

void BackwardAdd::compute_operands_grad(const Tensor& out) {
    m_operands[0].grad() += out.grad();
    m_operands[1].grad() += out.grad();
}

std::ostream& BackwardMinus::print(std::ostream& os) const {
    return os << "BackwardMinus";
}

void BackwardMinus::compute_operands_grad(const Tensor& out) {
    m_operands[0].grad() += -out.grad();
}

std::ostream& BackwardSub::print(std::ostream& os) const {
    return os << "BackwardSub";
}

void BackwardSub::compute_operands_grad(const Tensor& out) {
    m_operands[0].grad() += out.grad();
    m_operands[1].grad() += -out.grad();
}

std::ostream& BackwardMult::print(std::ostream& os) const {
    return os << "BackwardMult";
}

void BackwardMult::compute_operands_grad(const Tensor& out) {
    Tensor& a = m_operands[0];
    Tensor& b = m_operands[1];
    a.grad() += b * out.grad();
    b.grad() += a * out.grad();
}

std::ostream& BackwardPow::print(std::ostream& os) const {
    return os << "BackwardPow";
}

void BackwardPow::compute_operands_grad(const Tensor& out) {
    Tensor& base = m_operands[0];
    Tensor& exp = m_operands[1];
    Tensor ones(exp.shape());
    ones.fill_inplace(1.0f);
    
    base.grad() += (exp * base.pow(exp - ones)) * out.grad();
    
    Tensor log_base(base.m_node->log());
    exp.grad() += (base.pow(exp) * log_base) * out.grad();
}

std::ostream& BackwardLog::print(std::ostream& os) const {
    return os << "BackwardPow";
}

void BackwardLog::compute_operands_grad([[maybe_unused]] const Tensor& out) {
    // TensorNode& arg = *m_operands[0];
}
