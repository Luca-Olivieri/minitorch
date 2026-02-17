#include "compute.h"
#include "src/core/formatting.h"

using namespace mt::nn;

Linear::Linear(
    const size_t in_features,
    const size_t out_features,
    const bool bias

) {
    m_weight = Tensor({in_features, out_features}, 0.1f, true);
    if (bias) m_bias = Tensor({out_features}, 0.1f, true);
}

Tensor Linear::forward(
    const Tensor& input
) const {
    return input.matmul(m_weight);
}
