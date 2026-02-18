#include "activations.h"
#include "src/core/tensors.h"
#include "src/core/grad_fns.h"

using namespace mt::nn;

ReLU::ReLU() {}

Tensor ReLU::forward(
    const Tensor& input
) const {
    TensorStorage out_storage = input.m_node->m_storage.clone();
    for (size_t i {0}; i < out_storage.m_numel; ++i) {
        if (input.m_node->m_storage.get_entry_ref(i) < 0.0f) {
            out_storage.get_entry_ref(i) = 0.0f;
        }
    }
    std::shared_ptr<TensorNode> out = std::make_shared<TensorNode>(
        std::move(out_storage),
        input.compute_requires_grad_from_operands(std::vector<Tensor>{})
    );
    if (out->m_requires_grad) {
        out->m_grad_fn = std::make_unique<BackwardReLU>(
            input     // first operand
        );
    }
    return Tensor(out);
}
