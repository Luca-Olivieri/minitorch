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

void BackwardAdd::compute_operands_grad(
    const Tensor& out,
    bool create_graph)
{
    m_operands[0].accumulate_grad(out.grad(), create_graph);
    m_operands[1].accumulate_grad(out.grad(), create_graph);
}

std::ostream& BackwardMinus::print(std::ostream& os) const {
    return os << "BackwardMinus";
}

void BackwardMinus::compute_operands_grad(const Tensor& out, bool create_graph) {
    m_operands[0].accumulate_grad(-out.grad(), create_graph);
}

std::ostream& BackwardSub::print(std::ostream& os) const {
    return os << "BackwardSub";
}

void BackwardSub::compute_operands_grad(const Tensor& out, bool create_graph) {
    m_operands[0].accumulate_grad(out.grad(), create_graph);
    m_operands[1].accumulate_grad(-out.grad(), create_graph);
}

std::ostream& BackwardMult::print(std::ostream& os) const {
    return os << "BackwardMult";
}

void BackwardMult::compute_operands_grad(const Tensor& out, bool create_graph) {
    Tensor& a = m_operands[0];
    Tensor& b = m_operands[1];
    a.accumulate_grad(b * out.grad(), create_graph);
    b.accumulate_grad(a * out.grad(), create_graph);
}

std::ostream& BackwardDiv::print(std::ostream& os) const {
    return os << "BackwardDiv";
}

void BackwardDiv::compute_operands_grad(const Tensor& out, bool create_graph) {
    Tensor& a = m_operands[0];
    Tensor& b = m_operands[1];
    a.accumulate_grad(out.grad() / b, create_graph);
    b.accumulate_grad(-a / (b * b) * out.grad(), create_graph);
}

std::ostream& BackwardPow::print(std::ostream& os) const {
    return os << "BackwardPow";
}

void BackwardPow::compute_operands_grad(const Tensor& out, bool create_graph) {
    Tensor& base = m_operands[0];
    Tensor& exp = m_operands[1];
    Tensor ones(exp.shape());
    ones.fill_inplace(1.0f);
    
    base.accumulate_grad((exp * base.pow(exp - ones)) * out.grad(), create_graph);
    
    Tensor log_base(base.log());
    exp.accumulate_grad((base.pow(exp) * log_base) * out.grad(), create_graph);
}

std::ostream& BackwardLog::print(std::ostream& os) const {
    return os << "BackwardLog";
}

void BackwardLog::compute_operands_grad(
        const Tensor& out,
        bool create_graph)
{
    Tensor& x = m_operands[0];
    x.accumulate_grad(out.grad() / x, create_graph);
}

BackwardSum::BackwardSum(
    Tensor reduced_tensor,
    const size_t dim
):
    BackwardReduce(reduced_tensor),
    m_dim {dim} {}

std::ostream& BackwardSum::print(std::ostream& os) const {
    return os << "BackwardSum";
}

void BackwardSum::compute_operands_grad(
    [[maybe_unused]] const Tensor& out,
    [[maybe_unused]] bool create_graph
) {
    // Tensor& x = m_operands[0];
    // x.accumulate_grad(out.grad() / x, create_graph);
}

BackwardUnsqueeze::BackwardUnsqueeze(
    Tensor viewed_tensor,
    const size_t dim
):
    BackwardView(viewed_tensor),
    m_dim {dim} {}

std::ostream& BackwardUnsqueeze::print(std::ostream& os) const {
    return os << "BackwardUnsqueeze";
}

void BackwardUnsqueeze::compute_operands_grad(
    const Tensor& out,
    bool create_graph
) {
    Tensor& x = m_operands[0];
    x.accumulate_grad(out.grad().squeeze(m_dim), create_graph);
}

BackwardSqueeze::BackwardSqueeze(
    Tensor viewed_tensor,
    const size_t dim
):
    BackwardView(viewed_tensor),
    m_dim {dim} {}

std::ostream& BackwardSqueeze::print(std::ostream& os) const {
    return os << "BackwardSqueeze";
}

void BackwardSqueeze::compute_operands_grad(
    const Tensor& out,
    bool create_graph
) {
    Tensor& x = m_operands[0];
    x.accumulate_grad(out.grad().unsqueeze(m_dim), create_graph);
}
