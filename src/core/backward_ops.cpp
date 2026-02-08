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

void BackwardReshape::compute_operands_grad(const Tensor& out, bool create_graph) {
    std::vector<size_t> original_shape = m_operands[0].shape();
    m_operands[0].accumulate_grad(out.grad().reshape(original_shape), create_graph);
}

std::ostream& BackwardTranspose::print(std::ostream& os) const {
    return os << "BackwardTranspose";
}

BackwardTranspose::BackwardTranspose(
    Tensor viewed_tensor,
    size_t dim_1,
    size_t dim_2
):
    BackwardView { viewed_tensor },
    m_dim_1 { dim_1 },
    m_dim_2 { dim_2 } {}

void BackwardTranspose::compute_operands_grad(const Tensor& out, bool create_graph) {
    m_operands[0].accumulate_grad(out.grad().transpose(m_dim_2, m_dim_1), create_graph);
}

std::ostream& BackwardAdd::print(std::ostream& os) const {
    return os << "BackwardAdd";
}

void BackwardAdd::compute_operands_grad(const Tensor& out, bool create_graph) {
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

std::ostream& BackwardPow::print(std::ostream& os) const {
    return os << "BackwardPow";
}

void BackwardPow::compute_operands_grad(const Tensor& out, bool create_graph) {
    Tensor& base = m_operands[0];
    Tensor& exp = m_operands[1];
    Tensor ones(exp.shape());
    ones.fill_inplace(1.0f);
    
    base.accumulate_grad((exp * base.pow(exp - ones)) * out.grad(), create_graph);
    
    Tensor log_base(base.m_node->log());
    exp.accumulate_grad((base.pow(exp) * log_base) * out.grad(), create_graph);
}

std::ostream& BackwardLog::print(std::ostream& os) const {
    return os << "BackwardPow";
}

void BackwardLog::compute_operands_grad([[maybe_unused]] const Tensor& out, [[maybe_unused]] bool create_graph) {
    // TensorNode& arg = *m_operands[0];
}
