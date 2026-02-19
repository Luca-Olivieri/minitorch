#include "src/core/nn/optimizers.h"

Optimizer::Optimizer(
        std::map<std::string, Tensor>& parameters,
        float base_lr
):
    m_parameters{ parameters },
    m_base_lr{ base_lr } {}

void Optimizer::zero_grad() {
    for (auto& [name, tensor] : m_parameters) {
        if (tensor.m_node->m_grad) {
            tensor.reset_grad();
        }
    }
}

void SGD::step() {
    for (auto& [name, tensor] : m_parameters) {
        if (tensor.m_node->m_grad) {
            tensor -= tensor.grad() * m_base_lr;
        }
    }
}
