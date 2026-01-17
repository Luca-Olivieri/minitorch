#include <iostream>
#include <format>
#include <math.h>

#include "ops.h"

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
