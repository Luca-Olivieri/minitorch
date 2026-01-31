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

BackwardMult::BackwardMult(
    Tensor& t1,
    Tensor& t2
): m_t1(t1), m_t2(t2) {}

std::ostream& BackwardMult::print(std::ostream& os) const {
    return os << "BackwardMult";
}

std::ostream& operator<<(std::ostream& os, const BackwardMult& op) {
    return op.print(os);
}

void BackwardMult::backprop(Tensor& out) {
    for (size_t i = 0; i < m_t1.m_numel; i++) {
        size_t m_t1_lidx = m_t1.get_flat_index_from_logical(i);
        size_t m_t2_lidx = m_t2.get_flat_index_from_logical(i);
        size_t m_out_lidx = out.get_flat_index_from_logical(i);
        m_t1.m_flat_grad[m_t1_lidx] += m_t2.m_flat_data[m_t2_lidx] * out.m_flat_grad[m_out_lidx];
        m_t2.m_flat_grad[m_t2_lidx] += m_t1.m_flat_data[m_t1_lidx] * out.m_flat_grad[m_out_lidx];
    }
    m_t1.backprop();
    m_t2.backprop();
}

BackwardAdd::BackwardAdd(
    Tensor& t1,
    Tensor& t2
): m_t1(t1), m_t2(t2) {}

std::ostream& BackwardAdd::print(std::ostream& os) const {
    return os << "BackwardAdd";
}

std::ostream& operator<<(std::ostream& os, const BackwardAdd& op) {
    return op.print(os);
}

void BackwardAdd::backprop(Tensor& out) {
    for (size_t i = 0; i < m_t1.m_numel; i++) {
        size_t m_t1_lidx = m_t1.get_flat_index_from_logical(i);
        size_t m_t2_lidx = m_t2.get_flat_index_from_logical(i);
        size_t m_out_lidx = out.get_flat_index_from_logical(i);
        m_t1.m_flat_grad[m_t1_lidx] += out.m_flat_grad[m_out_lidx];
        m_t2.m_flat_grad[m_t2_lidx] += out.m_flat_grad[m_out_lidx];
    }
    m_t1.backprop();
    m_t2.backprop();
}

BackwardMinus::BackwardMinus(
    Tensor& t1
): m_t1(t1) {}

std::ostream& BackwardMinus::print(std::ostream& os) const {
    return os << "BackwardMinus";
}

std::ostream& operator<<(std::ostream& os, const BackwardMinus& op) {
    return op.print(os);
}

void BackwardMinus::backprop(Tensor& out) {
    for (size_t i = 0; i < m_t1.m_numel; i++) {
        size_t m_t1_lidx = m_t1.get_flat_index_from_logical(i);
        size_t m_out_lidx = out.get_flat_index_from_logical(i);
        m_t1.m_flat_grad[m_t1_lidx] += -out.m_flat_grad[m_out_lidx];
    }
    m_t1.backprop();
}

BackwardPow::BackwardPow(
    Tensor& t1,
    Tensor& t2
): m_t1(t1), m_t2(t2) {}

std::ostream& BackwardPow::print(std::ostream& os) const {
    return os << "BackwardPow";
}

std::ostream& operator<<(std::ostream& os, const BackwardPow& op) {
    return op.print(os);
}

void BackwardPow::backprop(Tensor& out) {
    for (size_t i = 0; i < m_t1.m_numel; i++) {
        size_t m_t1_lidx = m_t1.get_flat_index_from_logical(i);
        size_t m_t2_lidx = m_t2.get_flat_index_from_logical(i);
        size_t m_out_lidx = out.get_flat_index_from_logical(i);
        float base = m_t1.m_flat_data[m_t1_lidx];
        float exp = m_t2.m_flat_data[m_t2_lidx];
        m_t1.m_flat_grad[m_t1_lidx] += exp*std::pow(base, exp-1)*out.m_flat_grad[m_out_lidx];
    }
    m_t1.backprop();
}

// OLD OPS
// --- Node Implementation ---
Node::Node(){}

Node::Node(float value)
    : m_value(value) {}

void Node::backprop() {}

void Node::reset_grads() {
    m_grad = 0.0f;
}

std::ostream& operator<<(std::ostream& os, const Node& node){
    return os << std::format("Node(value={}, grad={})", node.m_value, node.m_grad);
}

// --- Operation implementation ---
void Operation::backward() {
    m_grad = 1.0f;
    backprop();
}

// --- UnaryOperation implementation ---
void UnaryOperation::reset_grads() {
    Node::reset_grads();
    m_x.reset_grads();
}

// --- BinaryOperation implementation ---
void BinaryOperation::reset_grads() {
    Node::reset_grads();
    m_x.reset_grads();
    m_y.reset_grads();
}

// NOTE: Do NOT use 'override' or 'virtual' in the .cpp file
// --- Addition Implementation ---
void Addition::forward() {
    m_value = m_x.m_value + m_y.m_value;
}

void Addition::backprop() {
    m_x.m_grad += m_grad;
    m_y.m_grad += m_grad;
    m_x.backprop();
    m_y.backprop();
}

// --- Multiplication Implementation ---
void Multiplication::forward() {
    m_value = m_x.m_value*m_y.m_value;
}

void Multiplication::backprop() {
    m_x.m_grad += m_y.m_value*m_grad;
    m_y.m_grad += m_x.m_value*m_grad;
    m_x.backprop();
    m_y.backprop();
}

// --- Power Implementation ---
void Squaration::forward() {
    m_value = m_x.m_value*m_x.m_value;
}

void Squaration::backprop() {
    m_x.m_grad += 2*m_x.m_value*m_grad;
    m_x.backprop();
}
