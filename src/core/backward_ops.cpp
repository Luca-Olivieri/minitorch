#include <iostream>
#include <format>
#include <math.h>
#include <memory>

#include "backward_ops.h"
#include "formatting.h"
#include "tensor_nodes.h"
#include "tensors.h"

GradFn::GradFn() {}

std::ostream& operator<<(std::ostream& os, const GradFn& op) {
    return op.print(os);
}

std::ostream& BackwardAdd::print(std::ostream& os) const {
    return os << "BackwardAdd";
}

void BackwardAdd::compute_operands_grad(
        const Tensor& out
) {
    m_operands[0].accumulate_grad(out.grad());
    m_operands[1].accumulate_grad(out.grad());
}

std::ostream& BackwardMinus::print(std::ostream& os) const {
    return os << "BackwardMinus";
}

void BackwardMinus::compute_operands_grad(
        const Tensor& out
) {
    m_operands[0].accumulate_grad(-out.grad());
}

std::ostream& BackwardSub::print(std::ostream& os) const {
    return os << "BackwardSub";
}

void BackwardSub::compute_operands_grad(
        const Tensor& out
) {
    m_operands[0].accumulate_grad(out.grad());
    m_operands[1].accumulate_grad(-out.grad());
}

std::ostream& BackwardMult::print(std::ostream& os) const {
    return os << "BackwardMult";
}

void BackwardMult::compute_operands_grad(
        const Tensor& out
) {
    Tensor& a = m_operands[0];
    Tensor& b = m_operands[1];
    a.accumulate_grad(b * out.grad());
    b.accumulate_grad(a * out.grad());
}

std::ostream& BackwardDiv::print(std::ostream& os) const {
    return os << "BackwardDiv";
}

void BackwardDiv::compute_operands_grad(
        const Tensor& out
) {
    Tensor& a = m_operands[0];
    Tensor& b = m_operands[1];
    a.accumulate_grad(out.grad() / b);
    b.accumulate_grad(-a / (b * b) * out.grad());
}

std::ostream& BackwardPow::print(std::ostream& os) const {
    return os << "BackwardPow";
}

void BackwardPow::compute_operands_grad(
        const Tensor& out
) {
    Tensor& base = m_operands[0];
    Tensor& exp = m_operands[1];
    Tensor ones(exp.shape());
    ones.fill_inplace(1.0f);
    
    base.accumulate_grad((exp * base.pow(exp - ones)) * out.grad());
    
    Tensor log_base(base.log());
    exp.accumulate_grad((base.pow(exp) * log_base) * out.grad());
}

std::ostream& BackwardLog::print(std::ostream& os) const {
    return os << "BackwardLog";
}

void BackwardLog::compute_operands_grad(
        const Tensor& out
) {
    Tensor& x = m_operands[0];
    x.accumulate_grad(out.grad() / x);
}

BackwardSum::BackwardSum(
        const Tensor reduced_tensor,
        const size_t dim,
        const size_t original_times
):
    BackwardReduce(reduced_tensor),
    m_dim {dim},
    m_original_times {original_times} {}

std::ostream& BackwardSum::print(std::ostream& os) const {
    return os << "BackwardSum";
}

void BackwardSum::compute_operands_grad(
        const Tensor& out
) {
    Tensor& x = m_operands[0];
    x.accumulate_grad(out.grad().unsqueeze(m_dim).repeat(m_dim, m_original_times));
}

BackwardUnsqueeze::BackwardUnsqueeze(
        const Tensor viewed_tensor,
        const size_t dim
):
    BackwardView(viewed_tensor),
    m_dim {dim} {}

std::ostream& BackwardUnsqueeze::print(std::ostream& os) const {
    return os << "BackwardUnsqueeze";
}

void BackwardUnsqueeze::compute_operands_grad(
        const Tensor& out
) {
    Tensor& x = m_operands[0];
    x.accumulate_grad(out.grad().squeeze(m_dim));
}

BackwardSqueeze::BackwardSqueeze(
        const Tensor viewed_tensor,
        const size_t dim
):
    BackwardView(viewed_tensor),
    m_dim {dim} {}

std::ostream& BackwardSqueeze::print(std::ostream& os) const {
    return os << "BackwardSqueeze";
}

void BackwardSqueeze::compute_operands_grad(
        const Tensor& out
) {
    Tensor& x = m_operands[0];
    x.accumulate_grad(out.grad().unsqueeze(m_dim));
}

BackwardRepeat::BackwardRepeat(
        const Tensor viewed_tensor,
        const size_t dim
):
    BackwardView(viewed_tensor),
    m_dim {dim} {}

std::ostream& BackwardRepeat::print(std::ostream& os) const {
    return os << "BackwardRepeat";
}

void BackwardRepeat::compute_operands_grad(
        const Tensor& out
) {
    Tensor& x = m_operands[0];
    x.accumulate_grad(out.grad().sum(m_dim).unsqueeze(m_dim));
}

BackwardClone::BackwardClone(
        const Tensor viewed_tensor
):
    BackwardView(viewed_tensor) {}

std::ostream& BackwardClone::print(std::ostream& os) const {
    return os << "BackwardClone";
}

void BackwardClone::compute_operands_grad(
        const Tensor& out
) {
    Tensor& x = m_operands[0];
    x.accumulate_grad(out.grad());
}
